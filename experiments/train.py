"""Train the class-balanced metric-learning and classifier experiments.

Examples:
  python -m experiments.train --config experiments/configs/phd08_metric.json --data-root /home/arseniy/data/PHD08_dataset
  python -m experiments.train --config experiments/configs/clinc150_metric.json --data-root /home/arseniy/data/clinc150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from experiments.data import DatasetCatalog, SampleDataset, SampleRef, collate_samples, load_catalog
from experiments.losses import CATMLLoss, ProxyAnchorLoss, multi_negative_ranking, multi_similarity, pair_contrastive, supervised_contrastive, triplet_hinge
from experiments.metrics import classification_metrics, open_set_metrics, retrieval_metrics
from experiments.models import build_encoder
from experiments.sampling import APMState, ClassGraph, PKBatchSampler, make_triplets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class ReferenceDataset(Dataset):
    def __init__(self, base: SampleDataset, refs: list[SampleRef]):
        self.base, self.refs = base, refs

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        return self.base[self.refs[index]]


def _tokenize(tokenizer, texts: list[str], device: torch.device, max_length: int):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _forward(model, batch: dict, modality: str, tokenizer, device: torch.device, max_length: int):
    if modality == "text":
        return model(_tokenize(tokenizer, batch["text"], device, max_length))
    images = batch["image"].to(device, non_blocking=True)
    if device.type == "cuda" and images.ndim == 4:
        images = images.contiguous(memory_format=torch.channels_last)
    return model(images)


def _class_ids(names: list[str], mapping: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([mapping[name] for name in names], device=device, dtype=torch.long)


def _make_objective(config: dict, train_count: int, embedding_dim: int, device: torch.device):
    loss_config = config.get("loss", {})
    name = loss_config.get("type", "catml")
    if name == "catml":
        objective = CATMLLoss(
            margin=loss_config.get("margin", 10.0),
            rho=loss_config.get("rho", 0.1),
            tau=loss_config.get("tau", 1.0),
            xi=loss_config.get("xi", 1.0),
            center_mode=loss_config.get("center_mode", "current"),
            class_weighting=loss_config.get("class_weighting", "per_sample"),
            norm_epsilon=loss_config.get("norm_epsilon", 0.0),
        )
    elif name == "proxy_anchor":
        objective = ProxyAnchorLoss(
            train_count,
            embedding_dim,
            margin=loss_config.get("margin", 0.1),
            alpha=loss_config.get("alpha", 32.0),
        )
    else:
        objective = None
    return name, objective.to(device) if objective is not None else None


def _batch_loss(loss_name, embeddings, labels, triples, loss_config, objective, train_label_ids):
    anchor, positive, negative = triples
    if loss_name == "catml":
        return objective(embeddings, labels, anchor, positive, negative)
    if loss_name == "triplet":
        return triplet_hinge(embeddings, anchor, positive, negative, margin=loss_config.get("margin", 0.2)), {}
    if loss_name == "contrastive":
        return pair_contrastive(embeddings, anchor, positive, negative, margin=loss_config.get("margin", 1.0)), {}
    if loss_name == "supcon":
        return supervised_contrastive(embeddings, labels, temperature=loss_config.get("temperature", 0.07)), {}
    if loss_name == "multiple_negatives":
        return multi_negative_ranking(embeddings, labels, temperature=loss_config.get("temperature", 0.05)), {}
    if loss_name == "multi_similarity":
        return multi_similarity(embeddings, labels), {}
    if loss_name == "proxy_anchor":
        local = torch.tensor([train_label_ids[int(x)] for x in labels.detach().cpu().tolist()], device=labels.device)
        return objective(embeddings, local), {}
    raise ValueError(f"Unsupported metric objective {loss_name!r}")


def _collect_embeddings(catalog, model, sample_dataset, labels, split, protocol, config, device, tokenizer):
    refs = list(catalog.references(labels, split, int(config["data_split_seed"]), protocol))
    if not refs:
        raise ValueError(f"No {split} examples for {catalog.name}; check the manifest and data root")
    return _encode_references(catalog, model, sample_dataset, refs, config, device, tokenizer)


def _encode_references(catalog, model, sample_dataset, refs, config, device, tokenizer):
    dataset = ReferenceDataset(sample_dataset, refs)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 512)),
        shuffle=False,
        num_workers=int(config["hardware"].get("num_workers", 8)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["hardware"].get("num_workers", 8)) > 0,
        collate_fn=collate_samples,
        worker_init_fn=worker_init_fn,
    )
    vectors, ids, uids = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and config["hardware"].get("precision", "bf16") == "bf16"):
                output = _forward(model, batch, catalog.modality, tokenizer, device, int(config["model"].get("max_length", 128)))
            vectors.append(output.float())
            ids.append(batch["label"].to(device, non_blocking=True))
            uids.extend(batch["uid"])
    return torch.cat(vectors), torch.cat(ids), uids


def _run_fewshot(config, section_name, catalog, model, sample_dataset, tokenizer, device, classes, protocol, artifact_dir=None):
    setting = config[section_name]
    n_way = int(setting.get("n_way", 5))
    k_shot = int(setting.get("k_shot", 1))
    query_per_class = int(setting.get("query_per_class", 5))
    episode_count = int(setting.get("episodes", 200))
    prediction = setting.get("prediction", "prototype")
    episode_seed = int(setting.get("episode_seed", config["data_split_seed"]))
    rng = random.Random(episode_seed + (0 if section_name == "validation" else 1_000_003))
    all_refs = []
    episode_indices = []
    episode_manifest = []
    for _ in range(episode_count):
        episode_labels = rng.sample(classes, n_way)
        support_map, query_map = {}, {}
        support_uids, query_uids = {}, {}
        for label in episode_labels:
            support = catalog.sample_class(label, setting.get("support_split", "gallery"), int(config["data_split_seed"]), protocol, k_shot, rng)
            query = catalog.sample_class(label, setting.get("query_split", "query"), int(config["data_split_seed"]), protocol, query_per_class, rng)
            support_map[label] = list(range(len(all_refs), len(all_refs) + len(support)))
            all_refs.extend(support)
            query_map[label] = list(range(len(all_refs), len(all_refs) + len(query)))
            all_refs.extend(query)
            support_uids[label] = [ref.uid for ref in support]
            query_uids[label] = [ref.uid for ref in query]
        episode_indices.append((episode_labels, support_map, query_map))
        episode_manifest.append({"classes": episode_labels, "support": support_uids, "query": query_uids})

    manifest_bytes = (json.dumps(episode_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if artifact_dir is not None:
        episode_path = artifact_dir / f"episodes_{section_name}.json"
        episode_path.write_bytes(manifest_bytes)

    vectors, _, _ = _encode_references(catalog, model, sample_dataset, all_refs, config, device, tokenizer)
    accuracies = []
    for episode_labels, support_map, query_map in episode_indices:
        support_vectors = torch.stack([vectors[index] for label in episode_labels for index in support_map[label]])
        support_labels = [label for label in episode_labels for _ in support_map[label]]
        query_indices = [index for label in episode_labels for index in query_map[label]]
        query_labels = [label for label in episode_labels for _ in query_map[label]]
        queries = vectors[query_indices]
        if prediction == "prototype":
            prototypes = torch.stack([vectors[support_map[label]].mean(dim=0) for label in episode_labels])
            scores = torch.nn.functional.normalize(queries.float(), dim=1) @ torch.nn.functional.normalize(prototypes.float(), dim=1).T
            predictions = [episode_labels[index] for index in scores.argmax(dim=1).cpu().tolist()]
        elif prediction == "nearest_support":
            distances = torch.cdist(queries.float(), support_vectors.float())
            predictions = [support_labels[index] for index in distances.argmin(dim=1).cpu().tolist()]
        else:
            raise ValueError("fewshot prediction must be prototype or nearest_support")
        accuracies.append(sum(pred == truth for pred, truth in zip(predictions, query_labels)) / max(1, len(query_labels)))
    return {
        "episode_accuracy": float(np.mean(accuracies)),
        "episode_std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "episodes": len(accuracies),
        "n_way": n_way,
        "k_shot": k_shot,
        "episode_manifest_sha256": manifest_hash,
    }


def _run_evaluation(config, section_name, catalog, model, head, sample_dataset, tokenizer, device, train_classes, protocol_manifest, artifact_dir=None):
    evaluation = config.get(section_name)
    if not evaluation:
        return {}
    protocol = evaluation.get("catalog_protocol", config.get("catalog_protocol", config["protocol"]))
    class_key = evaluation.get("class_key")
    evaluation_classes = evaluation.get("classes")
    if evaluation_classes is None and class_key:
        evaluation_classes = protocol_manifest.get(class_key)
    if evaluation_classes is None:
        evaluation_classes = train_classes
    if evaluation["task"] == "classification":
        refs = list(catalog.references(evaluation_classes, evaluation.get("split", "valid"), int(config["data_split_seed"]), protocol))
        dataset = ReferenceDataset(sample_dataset, refs)
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("evaluation", {}).get("batch_size", 512)),
            shuffle=False,
            num_workers=int(config["hardware"].get("num_workers", 8)),
            pin_memory=device.type == "cuda",
            persistent_workers=int(config["hardware"].get("num_workers", 8)) > 0,
            collate_fn=collate_samples,
            worker_init_fn=worker_init_fn,
        )
        y_true, y_pred = [], []
        train_local_id = {name: index for index, name in enumerate(train_classes)}
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and config["hardware"].get("precision", "bf16") == "bf16"):
                    embeddings = _forward(model, batch, catalog.modality, tokenizer, device, int(config["model"].get("max_length", 128)))
                    logits = head(embeddings)
                predicted = logits.argmax(dim=1).cpu().tolist()
                y_pred.extend(predicted)
                y_true.extend([train_local_id[name] for name in batch["class_name"]])
        return classification_metrics(y_true, y_pred)

    if evaluation["task"] == "retrieval":
        classes = evaluation_classes
        gallery, gallery_labels, gallery_ids = _collect_embeddings(
            catalog, model, sample_dataset, classes, evaluation.get("gallery_split", "gallery"), protocol, config, device, tokenizer
        )
        query, query_labels, query_ids = _collect_embeddings(
            catalog, model, sample_dataset, classes, evaluation.get("query_split", "query"), protocol, config, device, tokenizer
        )
        return retrieval_metrics(query, query_labels, gallery, gallery_labels, query_ids=query_ids,
                                 gallery_ids=gallery_ids, ks=tuple(evaluation.get("ks", [1, 2, 4, 8])))
    if evaluation["task"] == "fewshot":
        return _run_fewshot(config, section_name, catalog, model, sample_dataset, tokenizer, device,
                            evaluation_classes, protocol, artifact_dir=artifact_dir)
    if evaluation["task"] == "run_folder_oneshot":
        if catalog.modality != "image":
            raise ValueError("Omniglot run-folder one-shot evaluation only supports image encoders")
        from torchvision import transforms
        from oneshot import run_oneshot

        task_root = Path(evaluation["root"])
        transform = transforms.Compose([
            transforms.Resize((int(config["model"].get("image_size", 105)),) * 2),
            transforms.ToTensor(),
        ])
        runs = []
        model.eval()
        for run_folder in sorted(path for path in task_root.iterdir() if path.is_dir()):
            runs.append(float(run_oneshot(model, str(run_folder), transform)))
        if not runs:
            raise ValueError(f"No one-shot run folders found under {task_root}")
        return {"run_folder_oneshot_accuracy": float(np.mean(runs)),
                "run_folder_oneshot_std": float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0,
                "runs": len(runs)}
    if evaluation["task"] == "open_set":
        unknown_label = evaluation.get("unknown_label", "oos")
        known_classes = evaluation_classes
        gallery, gallery_labels, _ = _collect_embeddings(
            catalog, model, sample_dataset, known_classes, evaluation.get("gallery_split", "train"), protocol, config, device, tokenizer
        )
        prototypes = []
        for label in known_classes:
            local_id = catalog.labels.index(label)
            mask = gallery_labels == local_id
            if not mask.any():
                raise ValueError(f"No gallery samples for known class {label!r}")
            prototypes.append(gallery[mask].mean(dim=0))
        prototypes = torch.nn.functional.normalize(torch.stack(prototypes).float(), dim=1)
        known_refs = list(catalog.references(known_classes, evaluation.get("known_query_split", "valid"), int(config["data_split_seed"]), protocol))
        unknown_refs = list(catalog.references([unknown_label], evaluation.get("unknown_query_split", "oos_val"), int(config["data_split_seed"]), protocol))
        refs = known_refs + unknown_refs
        queries, query_ids, _ = _encode_references(catalog, model, sample_dataset, refs, config, device, tokenizer)
        confidence = (torch.nn.functional.normalize(queries.float(), dim=1) @ prototypes.T).max(dim=1).values.cpu().numpy()
        known_mask = np.asarray([reference.label != unknown_label for reference in refs], dtype=np.int32)
        threshold = evaluation.get("confidence_threshold")
        if section_name == "test" and threshold is None:
            threshold = config.get("validation", {}).get("calibrated_threshold")
            if threshold is None:
                raise ValueError("Open-set test needs a threshold saved/calibrated on validation")
        return open_set_metrics(known_mask, confidence, threshold=threshold)
    raise ValueError(f"Unsupported {section_name} task {evaluation['task']!r}")


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("dataset", "manifest", "protocol", "task", "model", "training", "sampler", "hardware")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Experiment config is missing keys: {missing}")
    loss_type = config.get("loss", {}).get("type", "catml")
    if config["task"] == "classification" and loss_type != "cross_entropy":
        raise ValueError("Classification configs must set loss.type='cross_entropy'")
    if config["task"] == "metric" and loss_type == "cross_entropy":
        raise ValueError("Metric configs need a metric objective, not cross_entropy")
    return config


def run(config_path: Path, data_root: Path, output_dir: Path, resume: Path | None = None, run_test: bool = False) -> None:
    config = _load_config(config_path)
    config["seed"] = int(config.get("seed", 20261113))
    set_seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by this experiment runner; select the Blackwell cluster environment")
    device = torch.device("cuda", int(config["hardware"].get("device", 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}; memory={props.total_memory / 2**30:.1f} GiB; capability={props.major}.{props.minor}")
    if props.major < 10:
        print("Warning: this config targets NVIDIA Blackwell (SM 10.x+); continuing on a different GPU.")

    manifest_path = config_path.parent / config["manifest"]
    if not manifest_path.exists():
        manifest_path = Path(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config["data_split_seed"] = int(config.get("data_split_seed", manifest.get("seed", config["seed"])))
    catalog = load_catalog(config["dataset"], data_root, hangul_only=bool(config.get("dataset_options", {}).get("hangul_only", True)))
    expected_label_hash = manifest.get("catalog_labels_sha256")
    if isinstance(expected_label_hash, dict):
        expected_label_hash = expected_label_hash["hangul_only" if config.get("dataset_options", {}).get("hangul_only", True) else "all_symbols"]
    if expected_label_hash:
        actual_label_hash = hashlib.sha256("\n".join(catalog.labels).encode("utf-8")).hexdigest()
        if actual_label_hash != expected_label_hash:
            raise ValueError("Dataset class inventory differs from the split manifest; regenerate the manifest before training")
    expected_fingerprints = dict(manifest.get("metadata_sha256", {}))
    if manifest.get("source_file") and manifest.get("source_sha256"):
        expected_fingerprints[manifest["source_file"]] = manifest["source_sha256"]
    for key in ("class_names_sha256", "alphabet_sha256", "class_folder_names_sha256"):
        if manifest.get(key):
            expected_fingerprints[key] = manifest[key]
    fingerprint_mismatch = {
        key: (expected, catalog.fingerprints.get(key))
        for key, expected in expected_fingerprints.items()
        if catalog.fingerprints.get(key) != expected
    }
    if fingerprint_mismatch:
        raise ValueError(f"Dataset metadata fingerprint mismatch; regenerate the manifest: {fingerprint_mismatch}")
    protocol_config = manifest["protocols"][config["protocol"]]
    train_class_key = config.get("train_class_key", "train_classes")
    train_classes = protocol_config.get(train_class_key)
    if train_classes is None and config["protocol"] == "closed_set":
        train_classes = protocol_config.get("classes")
    if not train_classes:
        raise ValueError(f"Manifest protocol {config['protocol']!r} has no {train_class_key!r}")
    missing_classes = sorted(set(train_classes) - set(catalog.labels))
    if missing_classes:
        raise ValueError(f"Manifest has classes not found in data; first missing: {missing_classes[:5]}")
    for section_name in ("validation", "test"):
        section = config.get(section_name)
        if not section or section.get("task") not in ("retrieval", "fewshot"):
            continue
        class_key = section.get("class_key")
        eval_classes = section.get("classes") or protocol_config.get(class_key) if class_key else section.get("classes")
        if eval_classes is not None:
            missing_eval = sorted(set(eval_classes) - set(catalog.labels))
            if missing_eval:
                raise ValueError(f"{section_name} manifest references classes absent from data: {missing_eval[:5]}")
            overlap = set(train_classes) & set(eval_classes)
            if overlap:
                raise ValueError(f"{section_name} class split overlaps training classes: {sorted(overlap)[:5]}")

    all_label_ids = {name: index for index, name in enumerate(catalog.labels)}
    train_label_ids = {name: index for index, name in enumerate(train_classes)}
    sampler_config = config["sampler"]
    apm = None
    if sampler_config.get("positive", "random") == "apm":
        apm = APMState(
            train_classes,
            gamma=sampler_config.get("apm_gamma", 1.0),
            smoothing=sampler_config.get("apm_smoothing", 0.0),
            unobserved_floor=sampler_config.get("apm_unobserved_floor", 0.01),
        )
    sampler = PKBatchSampler(
        catalog,
        train_classes,
        split=config["training"].get("train_split", "train"),
        protocol=config.get("catalog_protocol", config["protocol"]),
        classes_per_batch=int(sampler_config.get("classes_per_batch", 64)),
        samples_per_class=int(sampler_config.get("samples_per_class", 4)),
        steps_per_epoch=int(config["training"].get("steps_per_epoch", 100)),
        seed=config["seed"],
        data_split_seed=config["data_split_seed"],
        apm=apm,
        examples_per_class=config["training"].get("examples_per_class"),
    )
    sample_dataset = SampleDataset(
        catalog,
        all_label_ids,
        image_size=int(config["model"].get("image_size", 224)),
        train=True,
        image_channels=int(config["model"].get("image_channels", 3)),
        image_mean=config["model"].get("image_mean"),
        image_std=config["model"].get("image_std"),
        background_labels=train_classes,
    )
    evaluation_dataset = SampleDataset(
        catalog,
        all_label_ids,
        image_size=int(config["model"].get("image_size", 224)),
        train=False,
        image_channels=int(config["model"].get("image_channels", 3)),
        image_mean=config["model"].get("image_mean"),
        image_std=config["model"].get("image_std"),
    )
    workers = int(config["hardware"].get("num_workers", 8))
    worker_generator = torch.Generator(device="cpu")
    loader = DataLoader(
        sample_dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=bool(config["hardware"].get("persistent_workers", False)) and workers > 0,
        prefetch_factor=int(config["hardware"].get("prefetch_factor", 4)) if workers > 0 else None,
        collate_fn=collate_samples,
        worker_init_fn=worker_init_fn,
        generator=worker_generator,
    )

    model = build_encoder(config["model"]).to(device)
    if catalog.modality == "image" and config["hardware"].get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)
    classifier = None
    if config["task"] == "classification":
        classifier = nn.Linear(model.output_dim, len(train_classes)).to(device)
    loss_name, objective = _make_objective(config, len(train_classes), model.output_dim, device)
    optimizer_parameters = list(model.parameters())
    if classifier is not None:
        optimizer_parameters.extend(classifier.parameters())
    if objective is not None:
        optimizer_parameters.extend(objective.parameters())
    if not any(parameter.requires_grad for parameter in optimizer_parameters):
        raise ValueError("This config has no trainable parameters; use experiments.evaluate_pretrained for a frozen encoder")
    train_cfg = config["training"]
    try:
        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=float(train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=float(train_cfg.get("learning_rate", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg.get("epochs", 10)))
    class_graph = ClassGraph(len(catalog.labels), max_cluster_size=int(sampler_config.get("cluster_max_size", 200)))
    tokenizer = None
    if catalog.modality == "text":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"].get("model_name", "FacebookAI/roberta-base"),
            revision=config["model"].get("revision"),
            local_files_only=bool(config["model"].get("local_files_only", False)),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = dict(config)
    resolved_config["data_root"] = str(data_root.resolve())
    resolved_config["manifest"] = str(manifest_path.resolve())
    resolved_config["runtime"] = {
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": props.name,
        "gpu_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor],
        "python_seed": config["seed"],
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "label_map.json").write_text(json.dumps(all_label_ids, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best_score = -math.inf
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if classifier is not None and state.get("classifier"):
            classifier.load_state_dict(state["classifier"])
        if objective is not None and state.get("objective"):
            objective.load_state_dict(state["objective"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        best_score = float(state.get("best_score", -math.inf))
        if state.get("rng"):
            random.setstate(state["rng"]["python"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"])
            if state["rng"].get("cuda"):
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        if apm is not None and state.get("apm"):
            apm.load_state_dict(state["apm"])
        if state.get("class_graph"):
            class_graph.load_state_dict(state["class_graph"])
        sampler.epoch = int(state.get("sampler_epoch", state["epoch"] + 1))
        start_epoch = int(state["epoch"]) + 1
    else:
        start_epoch = 0

    log_path = output_dir / "metrics.jsonl"
    precision = config["hardware"].get("precision", "bf16")
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("hardware.precision must be bf16, fp16, or fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not report BF16 support; select fp16 or fp32 explicitly")
    amp_enabled = precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if resume is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])

    for epoch in range(start_epoch, int(train_cfg.get("epochs", 10))):
        sampler.set_epoch(epoch)
        # Non-persistent workers are re-seeded deterministically per epoch, so
        # stochastic transforms replay after a checkpoint resume.
        if not bool(config["hardware"].get("persistent_workers", False)):
            worker_generator.manual_seed(config["seed"] + epoch)
        model.train()
        if classifier is not None:
            classifier.train()
        if objective is not None:
            objective.train()
        epoch_loss = 0.0
        epoch_metrics: dict[str, float] = {}
        center_sum = torch.zeros((len(catalog.labels), model.output_dim), device=device)
        center_count = torch.zeros(len(catalog.labels), device=device)
        for step, batch in enumerate(loader):
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                embeddings = _forward(model, batch, catalog.modality, tokenizer, device, int(config["model"].get("max_length", 128)))
                if config["task"] == "classification":
                    local_targets = torch.tensor([train_label_ids[name] for name in batch["class_name"]], device=device)
                    logits = classifier(embeddings.float())
                    loss = nn.functional.cross_entropy(logits, local_targets)
                    batch_metrics = {"classification_loss": loss.detach()}
                else:
                    triples = make_triplets(
                        embeddings,
                        labels,
                        mode=sampler_config.get("negative", "random"),
                        margin=float(config.get("loss", {}).get("margin", 0.2)),
                        class_graph=class_graph,
                        directed_probability=float(sampler_config.get("directed_probability", 0.2)),
                    )
                    if loss_name == "proxy_anchor":
                        # Proxy Anchor uses normalized similarities; it has no triplet tuples.
                        local_targets = torch.tensor([train_label_ids[name] for name in batch["class_name"]], device=device)
                        loss = objective(embeddings, local_targets)
                        batch_metrics = {"proxy_anchor_loss": loss.detach()}
                    elif loss_name == "catml":
                        loss, batch_metrics = objective(embeddings, labels, *triples)
                    else:
                        loss, batch_metrics = _batch_loss(loss_name, embeddings, labels, triples, config.get("loss", {}), objective, train_label_ids)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().float().item())
            for key, value in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + float(value.detach().float().item())

            if apm is not None:
                local_labels = torch.tensor([train_label_ids[name] for name in batch["class_name"]], device=device)
                apm.update(embeddings.detach(), local_labels)
            center_sum.index_add_(0, labels, embeddings.detach().float())
            center_count.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

            cluster_every = int(sampler_config.get("cluster_update_steps", 0))
            if sampler_config.get("negative") == "auto_clusters" and cluster_every > 0 and (step + 1) % cluster_every == 0:
                active = torch.where(center_count > 0)[0]
                centers = center_sum / center_count.clamp_min(1).unsqueeze(1)
                class_graph.update(centers, active, fraction=float(sampler_config.get("cluster_fraction", 0.002)), max_edges=int(sampler_config.get("cluster_max_edges", 10000)))

        if sampler_config.get("negative") == "auto_clusters" and center_count.any():
            active = torch.where(center_count > 0)[0]
            centers = center_sum / center_count.clamp_min(1).unsqueeze(1)
            class_graph.update(centers, active, fraction=float(sampler_config.get("cluster_fraction", 0.002)), max_edges=int(sampler_config.get("cluster_max_edges", 10000)))
        scheduler.step()
        steps = max(1, len(loader))
        row = {"epoch": epoch, "train_loss": epoch_loss / steps, "lr": optimizer.param_groups[0]["lr"]}
        row.update({key: value / steps for key, value in epoch_metrics.items()})
        validation_cfg = config.get("validation")
        eval_interval = int(validation_cfg.get("every_n_epochs", 1)) if validation_cfg else 1
        validation_due = validation_cfg is not None and ((epoch + 1) % max(1, eval_interval) == 0 or epoch == int(train_cfg.get("epochs", 10)) - 1)
        validation = _run_evaluation(config, "validation", catalog, model, classifier, evaluation_dataset, tokenizer, device, train_classes, protocol_config, output_dir) if validation_due else {}
        if "confidence_threshold" in validation and config.get("validation") is not None:
            config["validation"]["calibrated_threshold"] = validation["confidence_threshold"]
        row["validation"] = validation
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False))

        primary = config.get("validation", {}).get("primary_metric", "recall@1" if config["task"] == "metric" else "accuracy")
        score = float(validation[primary]) if primary in validation else -row["train_loss"]
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "classifier": classifier.state_dict() if classifier is not None else None,
            "objective": objective.state_dict() if objective is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "class_labels": train_classes,
            "validation": validation,
            "apm": apm.state_dict() if apm is not None else None,
            "class_graph": class_graph.state_dict(),
            "sampler_epoch": epoch + 1,
            "best_score": max(best_score, score),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation_due and score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")
        elif not config.get("validation") and score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pt")
        if device.type == "cuda":
            print(f"peak_gpu_memory_gib={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")

    checkpoint_selection = config.get("test", {}).get("checkpoint", "best")
    selected_path = output_dir / ("last.pt" if checkpoint_selection == "last" else "best.pt")
    if run_test and selected_path.exists() and config.get("test"):
        best_state = torch.load(selected_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_state["model"])
        if classifier is not None and best_state.get("classifier"):
            classifier.load_state_dict(best_state["classifier"])
        if objective is not None and best_state.get("objective"):
            objective.load_state_dict(best_state["objective"])
        if best_state.get("validation", {}).get("confidence_threshold") is not None and config.get("validation") is not None:
            config["validation"]["calibrated_threshold"] = best_state["validation"]["confidence_threshold"]
        test_metrics = _run_evaluation(config, "test", catalog, model, classifier, evaluation_dataset, tokenizer, device, train_classes, protocol_config, output_dir)
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"test": test_metrics}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic metric-learning experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-test", action="store_true", help="Evaluate the configured test split once after training")
    args = parser.parse_args()
    run(args.config, args.data_root, args.output, args.resume, args.run_test)


if __name__ == "__main__":
    main()

