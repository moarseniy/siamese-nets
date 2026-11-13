"""Evaluate a pretrained encoder without training (e.g. frozen SentenceTransformer)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from dataset import ExperimentSampleDataset as SampleDataset, load_experiment_catalog as load_catalog
from model import build_experiment_encoder as build_encoder
from experiment_train import _load_config, _run_evaluation, set_seed


def evaluate(config_path: Path, data_root: Path, output_path: Path, split: str, calibrate_with_validation: bool = False):
    config = _load_config(config_path)
    config["seed"] = int(config.get("seed", 20261113))
    set_seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this evaluation runner")
    device = torch.device("cuda", int(config.get("hardware", {}).get("device", 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    manifest_path = (config_path.parent / config["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config["data_split_seed"] = int(config.get("data_split_seed", manifest.get("seed", config["seed"])))
    catalog = load_catalog(config["dataset"], data_root, hangul_only=bool(config.get("dataset_options", {}).get("hangul_only", True)))
    expected = manifest.get("catalog_labels_sha256")
    if isinstance(expected, dict):
        expected = expected["hangul_only" if config.get("dataset_options", {}).get("hangul_only", True) else "all_symbols"]
    if expected:
        actual = hashlib.sha256("\n".join(catalog.labels).encode("utf-8")).hexdigest()
        if actual != expected:
            raise ValueError("Dataset class inventory differs from the split manifest")
    protocol_manifest = manifest["protocols"][config["protocol"]]
    train_classes = protocol_manifest[config.get("train_class_key", "train_classes")]
    model = build_encoder(config["model"]).to(device).eval()
    sample_dataset = SampleDataset(
        catalog,
        {name: index for index, name in enumerate(catalog.labels)},
        image_size=int(config["model"].get("image_size", 224)),
        train=False,
        image_channels=int(config["model"].get("image_channels", 3)),
        image_mean=config["model"].get("image_mean"),
        image_std=config["model"].get("image_std"),
    )
    tokenizer = None
    if catalog.modality == "text":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["model_name"],
            revision=config["model"].get("revision"),
            local_files_only=bool(config["model"].get("local_files_only", False)),
        )

    if calibrate_with_validation and split == "test":
        validation = _run_evaluation(config, "validation", catalog, model, None, sample_dataset, tokenizer, device, train_classes, protocol_manifest)
        if "confidence_threshold" in validation:
            config.setdefault("validation", {})["calibrated_threshold"] = validation["confidence_threshold"]
    metrics = _run_evaluation(config, split, catalog, model, None, sample_dataset, tokenizer, device, train_classes, protocol_manifest, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"config": str(config_path.resolve()), "checkpoint": "pretrained-only", "split": split, "metrics": metrics}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
