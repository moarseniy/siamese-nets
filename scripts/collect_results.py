"""Export training and evaluation results to one CSV without touching running jobs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (REPO_ROOT / "experiments" / "runs", REPO_ROOT / "experiments" / "jobs")
BASE_FIELDS = (
    "source", "record_type", "split", "status", "epoch", "dataset", "protocol", "train_class_key",
    "seed", "loss", "batch_mode", "positive_sampler", "negative_sampler", "apm_gamma",
    "directed_probability", "cluster_fraction", "cluster_max_edges", "cluster_max_size",
    "cluster_update_steps", "make_cluster_on_epoch", "classes_per_batch", "samples_per_class",
    "backbone", "model_revision",
    "examples_per_class", "planned_epochs", "steps_per_epoch", "logical_batch_size", "batch_count",
    "minibatch_size", "mining_batch_size", "learning_rate", "precision",
    "gpu", "torch_version", "cuda_version", "primary_metric", "checkpoint",
)


def read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError, UnicodeError):
        # A process may be writing this file while the collector reads it.
        return None
    return value if isinstance(value, dict) else None


def flatten(prefix: str, values: dict) -> dict:
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten(name, value))
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, ensure_ascii=False)
        elif value is not None:
            result[name] = value
    return result


def source_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def metadata(config: dict) -> dict:
    model = config.get("model") or {}
    sampler = config.get("sampler") or {}
    training = config.get("training") or {}
    hardware = config.get("hardware") or {}
    runtime = config.get("runtime") or {}
    return {
        "dataset": config.get("dataset"),
        "protocol": config.get("protocol"),
        "train_class_key": config.get("train_class_key"),
        "seed": config.get("seed"),
        "loss": (config.get("loss") or {}).get("type"),
        "batch_mode": sampler.get("batch_mode", "pk"),
        "positive_sampler": sampler.get("positive"),
        "negative_sampler": sampler.get("negative"),
        "apm_gamma": sampler.get("apm_gamma"),
        "directed_probability": sampler.get("directed_probability"),
        "cluster_fraction": sampler.get("cluster_fraction"),
        "cluster_max_edges": sampler.get("cluster_max_edges"),
        "cluster_max_size": sampler.get("cluster_max_size"),
        "cluster_update_steps": sampler.get("cluster_update_steps"),
        "make_cluster_on_epoch": sampler.get("make_cluster_on_epoch"),
        "classes_per_batch": sampler.get("classes_per_batch"),
        "samples_per_class": sampler.get("samples_per_class"),
        "backbone": model.get("backbone") or model.get("model_name"),
        "model_revision": model.get("revision"),
        "examples_per_class": training.get("examples_per_class", "all"),
        "planned_epochs": training.get("epochs"),
        "steps_per_epoch": training.get("steps_per_epoch"),
        "logical_batch_size": training.get("batch_size"),
        "batch_count": training.get("batch_count"),
        "minibatch_size": training.get("minibatch_size"),
        "mining_batch_size": training.get("mining_batch_size"),
        "learning_rate": training.get("learning_rate"),
        "precision": hardware.get("precision"),
        "gpu": runtime.get("gpu_name"),
        "torch_version": runtime.get("torch"),
        "cuda_version": runtime.get("cuda_runtime"),
        "primary_metric": (config.get("validation") or {}).get("primary_metric"),
    }


def checkpoint_metadata(checkpoint: str | None) -> dict:
    if not checkpoint or checkpoint == "pretrained-only":
        return {}
    config = read_json(Path(checkpoint).parent / "resolved_config.json")
    if not config:
        return {}
    info = metadata(config)
    return {
        "training.seed": info["seed"], "loss": info["loss"],
        "batch_mode": info["batch_mode"],
        "training.batch_size": info["logical_batch_size"],
        "training.batch_count": info["batch_count"],
        "training.minibatch_size": info["minibatch_size"],
        "positive_sampler": info["positive_sampler"], "negative_sampler": info["negative_sampler"],
        "apm_gamma": info["apm_gamma"], "cluster_fraction": info["cluster_fraction"],
        "directed_probability": info["directed_probability"],
        "cluster_max_edges": info["cluster_max_edges"],
        "cluster_max_size": info["cluster_max_size"],
        "cluster_update_steps": info["cluster_update_steps"], "make_cluster_on_epoch": info["make_cluster_on_epoch"],
        "backbone": info["backbone"], "training.examples_per_class": info["examples_per_class"],
    }


def training_rows(config_path: Path) -> list[dict]:
    config = read_json(config_path)
    if config is None:
        return []
    run_dir = config_path.parent
    info = {"source": source_path(run_dir), **metadata(config)}
    rows = []
    log_path = run_dir / "metrics.jsonl"
    try:
        with log_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping incomplete/invalid line {line_number}: {log_path}", file=sys.stderr)
                    continue
                if not isinstance(row, dict):
                    continue
                item = {**info, "record_type": "epoch", "split": "validation", "epoch": row.get("epoch")}
                for key, value in row.items():
                    if key == "epoch":
                        continue
                    if key == "validation" and isinstance(value, dict):
                        item.update(flatten("validation", value))
                    elif isinstance(value, dict):
                        item.update(flatten(f"train.{key}", value))
                    else:
                        item[f"train.{key}"] = value
                rows.append(item)
    except FileNotFoundError:
        pass
    test = read_json(run_dir / "test_metrics.json")
    if test is not None:
        item = {**info, "record_type": "test", "split": "test", "checkpoint": (config.get("test") or {}).get("checkpoint", "best")}
        item.update(flatten("test", test))
        rows.append(item)
    return rows


def evaluation_row(path: Path) -> dict | None:
    result = read_json(path)
    if not result or not isinstance(result.get("metrics"), dict) or result.get("split") not in ("validation", "test"):
        return None
    config_path = Path(result.get("config", ""))
    config = read_json(config_path) if config_path.is_file() else None
    item = {
        "source": source_path(path), "record_type": "evaluation", "split": result["split"],
        **metadata(config or {}), **checkpoint_metadata(result.get("checkpoint")),
        "checkpoint": result.get("checkpoint"),
    }
    item.update(flatten(result["split"], result["metrics"]))
    return item


def batch_rows(path: Path) -> list[dict]:
    state = read_json(path)
    if not state or not isinstance(state.get("jobs"), list) or not isinstance(state.get("settings"), dict):
        return []
    settings = state["settings"]
    rows = []
    for job in state["jobs"]:
        if not isinstance(job, dict):
            continue
        command = job.get("command") or []
        config = {}
        if "--config" in command:
            index = command.index("--config")
            if index + 1 < len(command):
                config = read_json(Path(command[index + 1])) or {}
        checkpoint = None
        if "--checkpoint" in command:
            index = command.index("--checkpoint")
            if index + 1 < len(command):
                checkpoint = command[index + 1]
        attempts = job.get("attempts") or [{}]
        for attempt in attempts:
            source = attempt.get("output") or str(path.parent)
            error_tail = None
            if attempt.get("status") == "failed" and attempt.get("log"):
                try:
                    with Path(attempt["log"]).open("rb") as stream:
                        stream.seek(0, os.SEEK_END)
                        stream.seek(max(0, stream.tell() - 8192))
                        error_tail = "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-15:])
                except OSError:
                    pass
            row = {
                "source": source_path(Path(source)), "record_type": "batch_job",
                **metadata(config), **checkpoint_metadata(checkpoint), "checkpoint": checkpoint,
                "status": attempt.get("status", job.get("status")),
                "batch.name": settings.get("batch") or settings.get("series"), "batch.phase": settings.get("phase"),
                "batch.job_id": job.get("id"), "batch.attempt": attempt.get("number"),
                "batch.gpu": attempt.get("gpu"), "batch.started_at": attempt.get("started_at"),
                "batch.ended_at": attempt.get("ended_at"), "batch.returncode": attempt.get("returncode"),
                "batch.log": attempt.get("log"), "batch.error_tail": error_tail,
            }
            rows.append(row)
    return rows


def _collect(roots: list[Path], output: Path) -> int:
    rows = []
    seen_configs: set[Path] = set()
    seen_results: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for state_path in sorted(root.rglob("batch_state.json")):
            real_path = state_path.resolve()
            if real_path not in seen_results:
                seen_results.add(real_path)
                rows.extend(batch_rows(state_path))
        for config_path in sorted(root.rglob("resolved_config.json")):
            real_path = config_path.resolve()
            if real_path not in seen_configs:
                seen_configs.add(real_path)
                rows.extend(training_rows(config_path))
        for path in sorted(root.rglob("*.json")):
            if path.name in ("resolved_config.json", "test_metrics.json") or path.parent.name == "configs":
                continue
            real_path = path.resolve()
            if real_path in seen_results:
                continue
            seen_results.add(real_path)
            row = evaluation_row(path)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: (str(row["source"]), str(row["record_type"]), row.get("epoch") if isinstance(row.get("epoch"), int) else -1))
    fields = list(BASE_FIELDS) + sorted({key for row in rows for key in row} - set(BASE_FIELDS))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=output.parent, prefix=".results-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows)


def collect(roots: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Multiple schedulers and a live watcher may refresh the same report.
    # Serialize the entire snapshot so an older snapshot cannot overwrite a newer one.
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _collect(roots, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment results into one CSV")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments" / "results.csv")
    parser.add_argument("--search-root", type=Path, action="append", help="Also scan a custom results directory; may be repeated")
    parser.add_argument("--watch", action="store_true", help="Refresh periodically until Ctrl+C")
    parser.add_argument("--interval", type=float, default=60, help="Refresh interval in seconds for --watch")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    roots = [*DEFAULT_ROOTS, *(args.search_root or [])]
    try:
        while True:
            count = collect(roots, args.output)
            print(f"Collected {count} rows in {args.output}", flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped collecting results.", flush=True)


if __name__ == "__main__":
    main()
