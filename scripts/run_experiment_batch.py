"""Run resumable experiment batches for PHD08, Omniglot, CUB and CLINC150."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from run_phd08_batch import CONFIGS, ROOT, STATE_NAME, execute, now, prepare_matrix, save_state, summary


SERIES = {
    "p1": "phd08", "p2": "phd08", "p3": "phd08", "p4": "phd08", "p5": "phd08",
    "o1": "omniglot", "o2": "omniglot", "o3": "omniglot", "o4": "omniglot", "o5": "omniglot",
    "c1": "cub200", "c2": "cub200", "c3": "cub200", "c4": "cub200",
    "l1": "clinc150", "l2": "clinc150", "l3": "clinc150", "l4": "clinc150", "l5": "clinc150", "l6": "clinc150",
}
RETRIEVAL = {"o3", "c1", "l3"}
FEWSHOT = {"o4": "o3", "c4": "c1", "l4": "l3"}
CLOSED_FULL = {"o1", "c2", "l1"}
CLOSED_FEW = {"o2", "c3", "l2"}
BACKBONES = {"omniglot": ("resnet18",), "cub200": ("resnet50", "vit"), "clinc150": ("roberta", "minilm")}


def load_config(name: str) -> dict:
    path = CONFIGS / f"{name}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["manifest"] = str((path.parent / config["manifest"]).resolve())
    return config


def write_config(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def metric_base(dataset: str, backbone: str) -> dict:
    name = {
        ("omniglot", "resnet18"): "omniglot_metric_dev",
        ("cub200", "resnet50"): "cub200_metric_dev",
        ("cub200", "vit"): "cub200_vitb16_metric_dev",
        ("clinc150", "roberta"): "clinc150_metric_dev",
        ("clinc150", "minilm"): "clinc150_sentence_transformer_finetune_dev",
    }[(dataset, backbone)]
    return load_config(name)


def closed_templates(dataset: str, backbone: str) -> tuple[dict, dict]:
    ce = load_config(f"{dataset}_classification")
    metric_architecture = metric_base(dataset, backbone)
    # Use the same encoder and sampler in the CE and metric comparisons.
    ce["model"] = copy.deepcopy(metric_architecture["model"])
    ce["model"]["normalize"] = False
    ce["sampler"] = copy.deepcopy(metric_architecture["sampler"])
    ce["training"] = copy.deepcopy(metric_architecture["training"])
    ce["hardware"] = copy.deepcopy(metric_architecture["hardware"])
    ce["evaluation"] = copy.deepcopy(metric_architecture["evaluation"])
    for split in ("validation", "test"):
        ce[split].update(embedding_metrics=True, gallery_split="train", primary_metric="nearest_exemplar_accuracy")
    metric = copy.deepcopy(ce)
    metric["task"] = "metric"
    metric["loss"] = copy.deepcopy(metric_architecture["loss"])
    for split, query in (("validation", "valid"), ("test", "test")):
        metric[split] = {"task": "embedding_classification", "class_key": "classes", "gallery_split": "train",
                         "query_split": query, "primary_metric": "nearest_exemplar_accuracy"}
        if split == "test":
            metric[split]["checkpoint"] = "best"
    return metric, ce


def retrieval_templates(dataset: str, backbone: str, series: str) -> tuple[dict, dict]:
    metric = metric_base(dataset, backbone)
    if series == "l5":
        open_set = load_config("clinc150_open_set")
        for section in ("protocol", "catalog_protocol", "train_class_key", "validation", "test"):
            metric[section] = copy.deepcopy(open_set[section])
        metric["training"] = copy.deepcopy(open_set["training"])
        metric["evaluation"] = copy.deepcopy(open_set["evaluation"])
    ce = copy.deepcopy(metric)
    ce["classification_head"] = True
    ce["loss"] = {"type": "cross_entropy"}
    return metric, ce


def fewshot_jobs(settings: dict, batch_dir: Path) -> list[dict]:
    source = Path(settings["source_batch"])
    source_state = json.loads((source / STATE_NAME).read_text(encoding="utf-8"))
    source_settings = source_state.get("settings", {})
    expected_series = FEWSHOT[settings["series"]]
    source_series = (source_settings.get("series") or "").split(",")
    if expected_series not in source_series or source_settings.get("phase") != settings["phase"]:
        raise ValueError(f"{settings['series']} requires a {expected_series} batch with the same phase")
    if source_settings.get("data_root") != settings["data_root"]:
        raise ValueError("Few-shot data root must match the source training batch")
    if source_settings.get("backbone") != settings["backbone"]:
        raise ValueError("Few-shot backbone must match the source training batch")
    source_jobs = [job for job in source_state["jobs"] if job.get("id", "").startswith(f"{expected_series}__")]
    if not source_jobs or any(job.get("status") != "completed" for job in source_jobs):
        raise ValueError("Finish the source retrieval jobs before generating few-shot evaluations")
    dataset = settings["dataset"]
    template = load_config(f"{dataset}_fewshot_dev")
    split = "test" if settings["phase"] == "final" else "validation"
    jobs = []
    for source_job in source_jobs:
        attempt = Path(source_job["attempts"][-1]["output"])
        checkpoint = attempt / ("last.pt" if split == "test" else "best.pt")
        trained = json.loads((attempt / "resolved_config.json").read_text(encoding="utf-8"))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
        for shot in (1, 5):
            config = copy.deepcopy(template)
            config["model"] = copy.deepcopy(trained["model"])
            config["model"]["freeze_backbone"] = True
            config["manifest"] = trained["manifest"]
            config["train_class_key"] = trained["train_class_key"]
            config["data_split_seed"] = trained["data_split_seed"]
            config[split]["k_shot"] = shot
            config[split]["prediction"] = "both"
            config[split]["primary_prediction"] = "nearest_support"
            config[split]["primary_metric"] = "nearest_support_episode_accuracy"
            if dataset == "cub200" and split == "validation":
                config[split]["support_split"] = "fewshot_val_support"
                config[split]["query_split"] = "fewshot_val_query"
            # Evaluation must use the same embedding geometry as the source model.
            for key in ("distance", "embedding_transform"):
                value = (trained.get("evaluation") or {}).get(key)
                if value is not None:
                    config[split][key] = value
            config_path = write_config(batch_dir / "plans" / "eval_configs" / f"{source_job['id']}__k{shot}.json", config)
            command = [sys.executable, str(ROOT / "train.py"), "--config", str(config_path),
                       "--data-root", settings["data_root"], "--models-dir", settings["models_dir"],
                       "--evaluate-pretrained", "--split", split, "--checkpoint", str(checkpoint),
                       "--output", "OUTPUT_PLACEHOLDER"]
            jobs.append({"id": f"{source_job['id']}__k{shot}", "command": command,
                         "kind": "evaluation", "status": "pending", "attempts": []})
    return jobs


def prepare_oneshot(settings: dict, batch_dir: Path) -> list[dict]:
    root = Path(settings["oneshot_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Omniglot task folders not found: {root}")
    config = load_config("omniglot_run_folder_oneshot")
    config["test"]["root"] = str(root)
    config_path = write_config(batch_dir / "plans" / "oneshot_base.json", config)
    out = batch_dir / "plans" / "o5"
    command = [sys.executable, str(ROOT / "scripts" / "run_matrix.py"), "--config", str(config_path),
               "--data-root", settings["data_root"], "--models-dir", settings["models_dir"],
               "--output-dir", str(out), "--seeds", settings["seeds"],
               "--losses", "catml", "--selectors", "random", "--run-test"]
    if settings["epochs"] is not None:
        command += ["--epochs", str(settings["epochs"])]
    if settings["steps_per_epoch"] is not None:
        command += ["--steps-per-epoch", str(settings["steps_per_epoch"])]
    subprocess.run(command, cwd=ROOT, check=True)
    jobs = []
    for line in (out / "jobs.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        cmd = shlex.split(entry["command"])
        cmd[0] = sys.executable
        jobs.append({"id": f"o5__{Path(entry['config']).stem}", "command": cmd,
                     "kind": "training", "status": "pending", "attempts": []})
    return jobs


def prepare_series(settings: dict, batch_dir: Path, series: str) -> list[dict]:
    dataset, backbone, phase = (settings[k] for k in ("dataset", "backbone", "phase"))
    if series in FEWSHOT:
        single_settings = {**settings, "series": series}
        jobs = fewshot_jobs(single_settings, batch_dir)
    elif series == "o5":
        jobs = prepare_oneshot(settings, batch_dir)
    elif series == "l6":
        split = "test" if phase == "final" else "validation"
        command = [sys.executable, str(ROOT / "train.py"), "--config", str(CONFIGS / "clinc150_sentence_transformer_frozen.json"),
                   "--data-root", settings["data_root"], "--models-dir", settings["models_dir"],
                   "--evaluate-pretrained", "--split", split, "--output", "OUTPUT_PLACEHOLDER"]
        jobs = [{"id": f"l6_minilm_frozen_{split}", "command": command, "kind": "evaluation", "status": "pending", "attempts": []}]
    else:
        if series in CLOSED_FULL | CLOSED_FEW:
            metric, ce = closed_templates(dataset, backbone)
            budgets = "5,10" if series in CLOSED_FEW else "all"
            final_retrieval = False
        else:
            metric, ce = retrieval_templates(dataset, backbone, series)
            budgets = "all"
            final_retrieval = series in RETRIEVAL
        metric_path = write_config(batch_dir / "plans" / f"{series}_metric_base.json", metric)
        ce_path = write_config(batch_dir / "plans" / f"{series}_ce_base.json", ce)
        jobs = prepare_matrix(batch_dir / "plans" / series, metric_path, ce_path,
                              Path(settings["data_root"]), Path(settings["models_dir"]),
                              settings["seeds"], budgets, phase, settings["epochs"],
                              settings["steps_per_epoch"], final_retrieval=final_retrieval)
    return jobs


def prepare(settings: dict, batch_dir: Path) -> dict:
    jobs = []
    for series in settings["series"].split(","):
        jobs.extend(prepare_series(settings, batch_dir, series))
    return {"schema_version": 1, "created_at": now(), "settings": settings, "jobs": jobs}


def delegate_phd(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_phd08_batch.py"), "--batch", args.series,
           "--phase", args.phase, "--gpus", args.gpus, "--workers-per-gpu", str(args.workers_per_gpu),
           "--seeds", args.seeds]
    for flag, value in (("--phd-root", args.data_root), ("--models-dir", args.models_dir),
                        ("--kor-root", args.kor_root), ("--p3-dir", args.source_batch),
                        ("--output-dir", args.output_dir), ("--epochs", args.epochs),
                        ("--steps-per-epoch", args.steps_per_epoch)):
        if value is not None:
            cmd.extend((flag, str(value)))
    for flag, enabled in (("--dry-run", args.dry_run), ("--status", args.status), ("--retry-failed", args.retry_failed)):
        if enabled:
            cmd.append(flag)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", required=True, help="Series code or comma-separated series for one dataset")
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    parser.add_argument("--data-root", type=Path, help="Dataset root; P5 uses the PHD08 root here")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--kor-root", type=Path, help="HangulPrinted/KorSynthetic root for P5")
    parser.add_argument("--source-batch", type=Path, help="Completed P3/O3/C1/L3 batch for few-shot evaluation")
    parser.add_argument("--oneshot-root", type=Path, help="Omniglot all_runs task directory for O5")
    parser.add_argument("--backbone", help="Omniglot: resnet18; CUB: resnet50 or vit; CLINC: roberta or minilm")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    series_list = [value.strip().lower() for value in args.series.split(",") if value.strip()]
    if not series_list or len(set(series_list)) != len(series_list) or any(value not in SERIES for value in series_list):
        parser.error(f"--series must contain distinct codes from: {', '.join(SERIES)}")
    datasets = {SERIES[value] for value in series_list}
    if len(datasets) != 1:
        parser.error("A batch may combine series from one dataset; use separate batches for different data roots")
    args.series = ",".join(series_list)
    dataset = SERIES[series_list[0]]
    if dataset == "phd08":
        if len(series_list) != 1:
            parser.error("PHD08 series are run in separate batches")
        if args.backbone or args.oneshot_root:
            parser.error("PHD08 batches do not use --backbone or --oneshot-root")
        return delegate_phd(args)
    default_backbone = BACKBONES[dataset][0]
    backbone = args.backbone or ("minilm" if series_list == ["l6"] else default_backbone)
    if backbone not in BACKBONES[dataset] or (series_list == ["l6"] and backbone != "minilm"):
        parser.error(f"Invalid backbone for {args.series}; choose from {BACKBONES[dataset]}")
    batch_dir = (args.output_dir or ROOT / "experiments" / "jobs" / f"{dataset}_batch_{'_'.join(series_list)}_{args.phase}_{backbone}").expanduser().resolve()
    state_path = batch_dir / STATE_NAME
    if args.status:
        if not state_path.is_file():
            parser.error(f"Batch state does not exist: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(summary(state))
        for job in state["jobs"]:
            if job["status"] in ("failed", "interrupted") and job["attempts"]:
                print(f"{job['status']}: {job['id']} — {job['attempts'][-1]['log']}")
        return 0
    if args.data_root is None or args.models_dir is None:
        parser.error("--data-root and --models-dir are required")
    for path in (args.data_root, args.models_dir):
        if not path.expanduser().is_dir():
            parser.error(f"Directory not found: {path}")
    if any(series in FEWSHOT for series in series_list) and (len(series_list) != 1 or args.source_batch is None):
        parser.error("Few-shot series require their own batch and --source-batch")
    if "o5" in series_list and args.phase != "final":
        parser.error("O5 has only a test protocol; use --phase final after fixing settings")
    if len(series_list) == 1 and series_list[0] in (set(FEWSHOT) | {"l6"}) and (args.epochs is not None or args.steps_per_epoch is not None):
        parser.error("This series only evaluates models; training budget overrides do not apply")
    if args.workers_per_gpu < 1 or (args.epochs is not None and args.epochs < 1) or (args.steps_per_epoch is not None and args.steps_per_epoch < 1):
        parser.error("workers, epochs and steps must be positive")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    seeds = [value.strip() for value in args.seeds.split(",") if value.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain distinct GPU IDs")
    if not seeds or any(not value.isdigit() for value in seeds) or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain distinct non-negative integers")
    oneshot_root = args.oneshot_root
    if "o5" in series_list and oneshot_root is None:
        oneshot_root = next((p for p in (args.data_root / "python" / "all_runs", args.data_root / "all_runs") if p.is_dir()), None)
        if oneshot_root is None:
            parser.error("O5 needs --oneshot-root pointing to all_runs")
    settings = {
        "series": args.series, "dataset": dataset, "backbone": backbone, "phase": args.phase,
        "data_root": str(args.data_root.expanduser().resolve()), "models_dir": str(args.models_dir.expanduser().resolve()),
        "source_batch": str(args.source_batch.expanduser().resolve()) if args.source_batch is not None else None,
        "oneshot_root": str(oneshot_root.expanduser().resolve()) if oneshot_root is not None else None,
        "seeds": ",".join(seeds), "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
    }
    batch_dir.mkdir(parents=True, exist_ok=True)
    with (batch_dir / ".batch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another scheduler is already running this batch: {batch_dir}")
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("settings") != settings:
                parser.error("Saved batch has different parameters; use the original parameters or a new --output-dir")
        else:
            state = prepare(settings, batch_dir)
            save_state(state_path, state)
        print(f"Batch: {batch_dir}\n{summary(state)}", flush=True)
        if args.dry_run:
            return 0
        return execute(state, state_path, [gpu for gpu in gpus for _ in range(args.workers_per_gpu)], args.retry_failed)


if __name__ == "__main__":
    raise SystemExit(main())
