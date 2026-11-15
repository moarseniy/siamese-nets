"""Expand one base config into reproducible seed/loss/selector jobs.

This writes JSON jobs and a JSONL command list; it does not start training.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path


FINAL_PROTOCOLS = {
    "cub200": ("final_train_classes", "test_classes", "gallery", "query"),
    "omniglot": ("final_train_classes", "test_classes", "gallery", "query"),
    "phd08": ("final_train_classes", "test_classes", "gallery", "query"),
    "clinc150": ("final_train_classes", "test_classes", "gallery", "test_query"),
}

SELECTORS = {
    "random": ("random", "random"),
    "semi_hard": ("random", "semi_hard"),
    "batch_hard": ("random", "batch_hard"),
    "apm": ("apm", "random"),
    "auto_clusters": ("random", "auto_clusters"),
    "apm_auto": ("apm", "auto_clusters"),
}
LOSSES = {"catml", "triplet", "contrastive", "supcon", "multiple_negatives", "multi_similarity", "proxy_anchor", "cross_entropy"}


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an experiment job matrix")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cross-entropy-config", type=Path, help="Second base config used for the cross-entropy row in --phd08-baseline mode")
    parser.add_argument("--phd08-baseline", action="store_true", help="Use the fixed eight-row PHD08 comparison matrix")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, help="Second dataset root for cross-domain evaluation configs")
    parser.add_argument("--models-dir", type=Path, help="Local model repository folders and TorchVision weights")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--losses", default="catml,triplet")
    parser.add_argument("--selectors", default="random,apm_auto")
    parser.add_argument("--examples-per-class", default="all", help="Comma-separated train budgets: all,5,10")
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--contrastive-margin", type=float, default=1.0)
    parser.add_argument("--proxy-margin", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, help="Override epoch count for a short pilot")
    parser.add_argument("--steps-per-epoch", type=int, help="Override steps per epoch for a short pilot")
    parser.add_argument("--k-shot", type=int, help="Override K for few-shot validation and test episodes")
    parser.add_argument("--stage", choices=("development", "final"), default="development")
    parser.add_argument("--run-test", action="store_true", help="Include --run-test on generated commands")
    args = parser.parse_args()

    base = json.loads(args.config.read_text(encoding="utf-8"))
    cross_entropy_base = None
    if args.phd08_baseline:
        if args.cross_entropy_config is None:
            parser.error("--cross-entropy-config is required with --phd08-baseline")
        cross_entropy_base = json.loads(args.cross_entropy_config.read_text(encoding="utf-8"))
    elif args.cross_entropy_config is not None:
        parser.error("--cross-entropy-config requires --phd08-baseline")
    if args.models_dir is not None:
        args.models_dir = args.models_dir.expanduser().resolve()
        if not args.models_dir.is_dir():
            parser.error(f"models directory does not exist: {args.models_dir}")
    def prepare_base(config, path):
        if (
            config.get("evaluation_dataset")
            and config["evaluation_dataset"].get("data_root") is None
            and args.eval_data_root is None
        ):
            parser.error("--eval-data-root is required for configs with evaluation_dataset")
        config["manifest"] = str((path.parent / config["manifest"]).resolve())
        if config.get("evaluation_dataset", {}).get("manifest"):
            config["evaluation_dataset"]["manifest"] = str(
                (path.parent / config["evaluation_dataset"]["manifest"]).resolve()
            )
        return config

    base = prepare_base(base, args.config)
    if cross_entropy_base is not None:
        cross_entropy_base = prepare_base(cross_entropy_base, args.cross_entropy_config)
    if args.phd08_baseline:
        if base.get("dataset") not in ("phd08", "kor_synthetic") or base.get("task") != "metric":
            parser.error("--phd08-baseline needs a metric config for phd08 or kor_synthetic")
        if cross_entropy_base.get("dataset") != base.get("dataset"):
            parser.error("The metric and cross-entropy configs must use the same training dataset")
        if cross_entropy_base.get("loss", {}).get("type") != "cross_entropy":
            parser.error("The second config must set loss.type='cross_entropy'")
    losses, selectors = _csv(args.losses), _csv(args.selectors)
    budgets = _csv(args.examples_per_class)
    seeds = [int(value) for value in _csv(args.seeds)]
    if args.k_shot is not None and args.k_shot <= 0:
        raise ValueError("--k-shot must be positive")
    for budget in budgets:
        if budget != "all" and int(budget) <= 0:
            raise ValueError(f"Invalid examples-per-class budget {budget!r}")
    unknown_selectors = sorted(set(selectors) - set(SELECTORS))
    if unknown_selectors:
        raise ValueError(f"Unknown selectors {unknown_selectors}; choose from {sorted(SELECTORS)}")
    unknown_losses = sorted(set(losses) - LOSSES)
    if unknown_losses:
        raise ValueError(f"Unknown losses {unknown_losses}; choose from {sorted(LOSSES)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / "configs"
    run_dir = args.output_dir / "runs"
    config_dir.mkdir(exist_ok=True)
    jobs = []
    if args.phd08_baseline:
        comparisons = [
            ("catml", "random"), ("catml", "apm"), ("catml", "auto_clusters"), ("catml", "apm_auto"),
            ("triplet", "random"), ("triplet", "semi_hard"), ("triplet", "batch_hard"),
            ("cross_entropy", None),
        ]
    else:
        comparisons = [(loss_name, selector) for loss_name in losses for selector in selectors]
    for seed in seeds:
        for loss_name, selector in comparisons:
            for budget in budgets:
                job_base = cross_entropy_base if loss_name == "cross_entropy" and cross_entropy_base is not None else base
                config = copy.deepcopy(job_base)
                if config["task"] == "classification" and loss_name != "cross_entropy":
                    raise ValueError("For a classification config, --losses must be cross_entropy")
                if config["task"] == "metric" and loss_name == "cross_entropy" and not config.get("classification_head", False):
                    raise ValueError("Metric-task cross-entropy config needs classification_head=true")
                config["seed"] = seed
                config["loss"]["type"] = loss_name
                if loss_name == "triplet":
                    config["loss"]["margin"] = args.triplet_margin
                elif loss_name == "contrastive":
                    config["loss"]["margin"] = args.contrastive_margin
                elif loss_name == "proxy_anchor":
                    config["loss"]["margin"] = args.proxy_margin
                    config["loss"].setdefault("alpha", 32.0)
                elif loss_name == "supcon":
                    config["loss"].setdefault("temperature", 0.07)
                if selector is not None:
                    config["sampler"]["positive"], config["sampler"]["negative"] = SELECTORS[selector]
                if selector == "apm":
                    config["sampler"]["apm_gamma"] = 1.0
                elif selector == "apm_auto":
                    config["sampler"]["apm_gamma"] = 2.0
                config["training"]["examples_per_class"] = None if budget == "all" else int(budget)
                if budget != "all":
                    config["sampler"]["samples_per_class"] = min(
                        int(config["sampler"]["samples_per_class"]), int(budget)
                    )
                if args.epochs is not None:
                    config["training"]["epochs"] = args.epochs
                if args.steps_per_epoch is not None:
                    config["training"]["steps_per_epoch"] = args.steps_per_epoch
                if args.k_shot is not None:
                    fewshot_sections = [
                        section for section in ("validation", "test")
                        if config.get(section, {}).get("task") == "fewshot"
                    ]
                    if not fewshot_sections:
                        raise ValueError("--k-shot can only be used with few-shot evaluation configs")
                    for section in fewshot_sections:
                        config[section]["k_shot"] = args.k_shot
                if selector in ("auto_clusters", "apm_auto"):
                    config["sampler"].setdefault("cluster_update_steps", 0)
                    config["sampler"].setdefault("cluster_fraction", 0.002)
                if args.stage == "final" and config["task"] == "metric":
                    if config.get("validation", {}).get("task") != "retrieval":
                        raise ValueError("Final matrix generation is only defined for retrieval dev configs; use an explicit final config for few-shot/OOS protocols")
                    dataset = config["dataset"]
                    if dataset not in FINAL_PROTOCOLS:
                        raise ValueError(f"No final-stage mapping for dataset {dataset!r}")
                    train_key, test_key, gallery_split, query_split = FINAL_PROTOCOLS[dataset]
                    config["train_class_key"] = train_key
                    config.pop("validation", None)
                    config["test"] = {
                        "task": "retrieval",
                        "class_key": test_key,
                        "catalog_protocol": config.get("catalog_protocol", config["protocol"]),
                        "gallery_split": gallery_split,
                        "query_split": query_split,
                        "ks": [1, 2, 4, 8],
                        "checkpoint": "last",
                        "distance": config.get("validation", {}).get("distance", config.get("evaluation", {}).get("distance", "cosine")),
                        "embedding_transform": config.get("validation", {}).get("embedding_transform", config.get("evaluation", {}).get("embedding_transform", "identity")),
                    }
                elif args.stage == "final":
                    config.pop("validation", None)
                    if config.get("test"):
                        config["test"]["checkpoint"] = "last"
                selector_name = selector or "standard"
                name = f"{config['dataset']}_{args.stage}_{loss_name}_{selector_name}_{budget}_seed{seed}.json"
                config_path = config_dir / name
                config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                output_path = run_dir / name.removesuffix(".json")
                command = ["python", "train.py", "--config", str(config_path), "--data-root", str(args.data_root), "--output", str(output_path)]
                if args.eval_data_root is not None:
                    command.extend(("--eval-data-root", str(args.eval_data_root)))
                if args.models_dir is not None:
                    command.extend(("--models-dir", str(args.models_dir)))
                if args.run_test and config.get("test"):
                    command.append("--run-test")
                jobs.append({"config": str(config_path), "output": str(output_path), "command": shlex.join(command)})
    with (args.output_dir / "jobs.jsonl").open("w", encoding="utf-8") as stream:
        for job in jobs:
            stream.write(json.dumps(job, ensure_ascii=False) + "\n")
    script = args.output_dir / "run.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(job["command"] for job in jobs) + "\n", encoding="utf-8")
    script.chmod(0o755)
    print(f"Wrote {len(jobs)} jobs under {args.output_dir}; no training was started.")


if __name__ == "__main__":
    main()

