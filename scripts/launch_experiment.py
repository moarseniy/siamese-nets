"""One command to launch a configured experiment or a generated run matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
DEFAULT_DATA_ROOTS = {
    "clinc150": Path("/home/arseniy/data/clinc150"),
    "cub200": Path("/home/arseniy/data/CUB-200-2011"),
    "omniglot": Path("/home/arseniy/data/omniglot"),
    "phd08": Path("/home/arseniy/data/PHD08_dataset"),
    "kor_synthetic": Path("/home/arseniy/data/kor_synthetic"),
}


def _repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _config_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_absolute():
        if candidate.suffix != ".json":
            candidate = candidate.with_suffix(".json")
    elif candidate.parent == Path("."):
        candidate = CONFIG_ROOT / (candidate.name if candidate.suffix == ".json" else f"{candidate.name}.json")
    else:
        candidate = REPO_ROOT / candidate
        if candidate.suffix != ".json":
            candidate = candidate.with_suffix(".json")
    if candidate.is_file():
        return candidate.resolve()
    choices = ", ".join(sorted(path.stem for path in CONFIG_ROOT.glob("*.json")))
    raise FileNotFoundError(f"Experiment config not found: {value}. Available: {choices}")


def _default_selector(config: dict) -> str:
    sampler = config.get("sampler", {})
    positive = sampler.get("positive", "random")
    negative = sampler.get("negative", "random")
    if positive == "apm" and negative == "auto_clusters":
        return "apm_auto"
    if positive == "apm":
        return "apm"
    if negative in ("auto_clusters", "semi_hard", "batch_hard"):
        return negative
    return "random"


def _dataset_root(dataset: str, override: Path | None) -> Path:
    root = _repo_path(override) if override is not None else DEFAULT_DATA_ROOTS.get(dataset)
    if root is None:
        raise ValueError(f"No default data root for dataset {dataset!r}; pass --data-root")
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}; pass --data-root to override it")
    return root.resolve()


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Config file path or name from experiments/configs")
    parser.add_argument("--models-dir", type=Path, required=True, help="Local pretrained models root")
    parser.add_argument("--data-root", type=Path, help="Override the default dataset root")
    parser.add_argument("--eval-data-root", type=Path, help="Override the second dataset root, used by P5")
    parser.add_argument("--output", type=Path, help="Output directory/file for a single run")
    parser.add_argument("--output-dir", type=Path, help="Matrix jobs directory")
    parser.add_argument("--seed", type=int, help="Run one seed instead of the seed in the config")
    parser.add_argument("--seeds", help="Comma-separated seeds; starts a matrix")
    parser.add_argument("--losses", help="Comma-separated objectives; starts a matrix")
    parser.add_argument("--selectors", help="Comma-separated selectors; starts a matrix")
    parser.add_argument("--examples-per-class", help="Comma-separated budgets such as all,5,10")
    parser.add_argument("--k-shot", type=int, help="Few-shot K override")
    parser.add_argument("--epochs", type=int, help="Epoch override for a pilot/matrix")
    parser.add_argument("--steps-per-epoch", type=int, help="Step override for a pilot/matrix")
    parser.add_argument("--stage", choices=("development", "final"), help="Use a final class-disjoint retrieval protocol")
    parser.add_argument("--run-test", action="store_true", help="Run the configured test after training")
    parser.add_argument("--resume", type=Path, help="Resume one run from a checkpoint")
    parser.add_argument("--split", choices=("validation", "test"), default="validation", help="Split for inference-only configs")
    parser.add_argument("--calibrate-with-validation", action="store_true", help="Calibrate an open-set threshold before test")
    parser.add_argument("--dry-run", action="store_true", help="Prepare/print launch commands without training")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one experiment or its run matrix through a consistent interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(parser)
    args = parser.parse_args()

    config_path = _config_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "dataset" not in config or "manifest" not in config:
        parser.error("launch_experiment.py accepts manifest configs from experiments/configs")

    models_dir = _repo_path(args.models_dir)
    if not models_dir.is_dir():
        parser.error(f"models directory does not exist: {models_dir}")
    data_root = _dataset_root(config["dataset"], args.data_root)
    eval_data_root = None
    if config.get("evaluation_dataset"):
        eval_dataset_name = config["evaluation_dataset"]["dataset"]
        eval_data_root = _dataset_root(eval_dataset_name, args.eval_data_root)
    elif args.eval_data_root is not None:
        parser.error("--eval-data-root is only used by a config with evaluation_dataset")

    model_config = config.get("model", {})
    inference_only = bool(model_config.get("freeze_backbone", False))
    matrix_mode = any((
        args.seed is not None,
        args.seeds is not None,
        args.losses is not None,
        args.selectors is not None,
        args.examples_per_class is not None,
        args.k_shot is not None,
        args.epochs is not None,
        args.steps_per_epoch is not None,
        args.stage is not None,
    ))
    if args.seed is not None and args.seeds is not None:
        parser.error("use either --seed or --seeds, not both")
    if args.output is not None and matrix_mode:
        parser.error("--output is for one run; use --output-dir for a matrix")
    if args.output_dir is not None and not matrix_mode:
        parser.error("--output-dir is for a matrix; use --output for one run")
    if args.resume is not None and matrix_mode:
        parser.error("--resume is for one run; resume each generated job separately")
    if inference_only and matrix_mode:
        parser.error("inference-only configs cannot be expanded into training matrices")
    if inference_only and args.run_test:
        parser.error("inference-only configs use --split test, not --run-test")
    if inference_only and args.resume is not None:
        parser.error("inference-only configs do not use checkpoints")
    if not inference_only and args.split != "validation":
        parser.error("--split is only for inference-only configs; use --run-test after training")
    if args.calibrate_with_validation and (not inference_only or args.split != "test"):
        parser.error("--calibrate-with-validation requires an inference-only config with --split test")
    if args.run_test and not inference_only and not config.get("test") and args.stage != "final":
        parser.error("this config has no test section; use --stage final for class-disjoint retrieval")
    if args.stage == "final" and config.get("validation", {}).get("task") != "retrieval":
        parser.error("--stage final is only for metric configs with retrieval validation")

    if matrix_mode:
        default_stage = args.stage or "development"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _repo_path(args.output_dir) if args.output_dir else REPO_ROOT / "experiments" / "jobs" / f"{config_path.stem}_{default_stage}_{stamp}"
        seed_values = args.seeds or str(args.seed if args.seed is not None else config.get("seed", 20261113))
        loss_values = args.losses or config.get("loss", {}).get("type", "catml")
        selector_values = args.selectors or _default_selector(config)
        budget_values = args.examples_per_class or "all"
        command = [
            sys.executable, str(REPO_ROOT / "scripts" / "run_matrix.py"),
            "--config", str(config_path),
            "--data-root", str(data_root),
            "--models-dir", str(models_dir),
            "--output-dir", str(output_dir),
            "--seeds", seed_values,
            "--losses", loss_values,
            "--selectors", selector_values,
            "--examples-per-class", budget_values,
            "--stage", default_stage,
        ]
        if eval_data_root is not None:
            command.extend(("--eval-data-root", str(eval_data_root)))
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        if args.steps_per_epoch is not None:
            command.extend(("--steps-per-epoch", str(args.steps_per_epoch)))
        if args.k_shot is not None:
            command.extend(("--k-shot", str(args.k_shot)))
        if args.run_test:
            command.append("--run-test")

        print("Preparing and launching matrix:", shlex.join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        run_script = output_dir / "run.sh"
        if args.dry_run:
            print(f"Dry run only. To launch later: bash {run_script}")
            return
        subprocess.run(["bash", str(run_script)], cwd=REPO_ROOT, check=True)
        return

    if args.run_test and not config.get("test"):
        parser.error("this config has no test section")
    if inference_only:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _repo_path(args.output) if args.output else (
            REPO_ROOT / "experiments" / "runs" / config_path.stem / f"{args.split}_{stamp}.json"
        )
        command = [
            sys.executable, str(REPO_ROOT / "train.py"),
            "--config", str(config_path), "--data-root", str(data_root),
            "--models-dir", str(models_dir), "--evaluate-pretrained",
            "--split", args.split, "--output", str(output_path),
        ]
        if args.calibrate_with_validation:
            command.append("--calibrate-with-validation")
    else:
        seed = int(config.get("seed", 20261113))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _repo_path(args.output) if args.output else (
            REPO_ROOT / "experiments" / "runs" / config_path.stem / f"seed{seed}_{stamp}"
        )
        command = [
            sys.executable, str(REPO_ROOT / "train.py"),
            "--config", str(config_path), "--data-root", str(data_root),
            "--models-dir", str(models_dir), "--output", str(output_dir),
        ]
        if eval_data_root is not None:
            command.extend(("--eval-data-root", str(eval_data_root)))
        if args.resume is not None:
            command.extend(("--resume", str(_repo_path(args.resume))))
        if args.run_test:
            command.append("--run-test")

    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
