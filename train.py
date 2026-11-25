"""Canonical command line for manifest-based PhD experiments."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


def check_runtime() -> None:
    expected = {"torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130"}
    problems = []
    if sys.version_info[:2] != (3, 13):
        problems.append(f"Python 3.13 required; found {sys.version_info.major}.{sys.version_info.minor}")
    for package, required_version in expected.items():
        try:
            actual_version = version(package)
        except PackageNotFoundError:
            problems.append(f"{package} is not installed")
            continue
        if actual_version != required_version:
            problems.append(f"{package} {required_version} required; found {actual_version}")
    if problems:
        details = "; ".join(problems)
        raise SystemExit(f"Unsupported experiment environment: {details}. Install the pinned requirements.txt.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Manifest experiment JSON")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root")
    parser.add_argument("--eval-data-root", type=Path, help="Second dataset root for cross-domain evaluation")
    parser.add_argument("--models-dir", type=Path, help="Local pretrained model assets")
    parser.add_argument("--output", type=Path, required=True, help="Unique run directory or evaluation result file")
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume training from")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint for pretrained evaluation")
    parser.add_argument("--run-test", action="store_true", help="Run the fixed test protocol after training")
    parser.add_argument("--evaluate-pretrained", action="store_true", help="Evaluate without training")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--calibrate-with-validation", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"config file does not exist: {config_path}")
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"dataset root does not exist: {data_root}")
    check_runtime()

    if args.evaluate_pretrained:
        if args.run_test or args.resume:
            parser.error("--evaluate-pretrained cannot be combined with --run-test or --resume")
        from experiment_evaluate_pretrained import evaluate

        evaluate(
            config_path,
            data_root,
            args.output.expanduser().resolve(),
            args.split,
            args.calibrate_with_validation,
            args.models_dir,
            args.checkpoint,
        )
        return

    if args.checkpoint or args.calibrate_with_validation or args.split != "validation":
        parser.error("--checkpoint, --calibrate-with-validation, and --split apply only to pretrained evaluation")
    from experiment_train import run

    run(
        config_path,
        data_root,
        args.output.expanduser().resolve(),
        args.resume.expanduser().resolve() if args.resume else None,
        args.run_test,
        args.eval_data_root.expanduser().resolve() if args.eval_data_root else None,
        args.models_dir.expanduser().resolve() if args.models_dir else None,
    )


if __name__ == "__main__":
    main()
