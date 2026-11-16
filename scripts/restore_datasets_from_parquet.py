"""Restore Parquet-converted PHD08/KorSynthetic catalogs to their folder layouts."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq


FORMAT = "siamese-nets-parquet-v1"
STATE_DIRNAME = ".restore_state"
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _restore_class(task: tuple) -> tuple[int, int]:
    dataset, class_index, source_root, class_root, relative_files, resume = task
    class_root = Path(class_root)
    source_root = Path(source_root)
    output_class = class_root / str(class_index)
    restored = 0
    for relative in relative_files:
        parquet_file = pq.ParquetFile(source_root / relative)
        for row_group in range(parquet_file.metadata.num_row_groups):
            for row in parquet_file.read_row_group(row_group).to_pylist():
                filename = str(row["filename"])
                target = output_class / filename
                if target.exists():
                    if resume:
                        continue
                    raise FileExistsError(f"Restored sample already exists: {target}; use --resume")

                if dataset == "phd08":
                    image = row["image"]
                    content = image.get("bytes") if isinstance(image, dict) else image
                    if content is None:
                        raise ValueError(f"Parquet image bytes are missing for class {class_index}/{filename}")
                    _atomic_write(target, lambda temporary, data=content: temporary.write_bytes(data))
                else:
                    if "source_json" in row:
                        content = row["source_json"]
                        _atomic_write(target, lambda temporary, data=content: temporary.write_bytes(data))
                    else:
                        background = [float(value) for value in row["background"]]
                        mask = [float(value) for value in row["mask"]]
                        shape = [int(value) for value in row.get("shape", [37, 37, 1])]
                        payload = [
                            {"data": background, "label": row.get("background_record", "main_data"), "shape": shape},
                            {"data": mask, "label": row.get("mask_record", "main_data_mask"), "shape": shape},
                            {"data": [float(row.get("markup_source", 0.0))], "label": "markup_source", "shape": [1, 1, 1]},
                            {"data": [float(row["symbol_index"])], "label": "symbols_0", "shape": [1, 1, 1]},
                        ]
                        _atomic_write(
                            target,
                            lambda temporary, value=payload: temporary.write_text(
                                json.dumps(value, ensure_ascii=False), encoding="utf-8"
                            ),
                        )
                restored += 1
    return class_index, restored


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore source-layout folders from a converted Parquet dataset")
    parser.add_argument("--dataset", choices=("phd08", "kor_synthetic"), required=True)
    parser.add_argument("--parquet-root", type=Path, required=True, help="Converted dataset root with dataset_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="New root for the restored folder dataset")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted restoration")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    source_root = args.parquet_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source_root == output or source_root in output.parents or output in source_root.parents:
        parser.error("Use separate source and output directories")
    manifest_path = source_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        parser.error(f"Missing Parquet dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("dataset") != args.dataset:
        parser.error("Parquet manifest format/dataset does not match --dataset")
    class_files = manifest.get("class_files")
    if not isinstance(class_files, dict) or not class_files:
        parser.error("Parquet manifest has no class_files mapping")

    output.mkdir(parents=True, exist_ok=True)
    state_dir = output / STATE_DIRNAME
    state_dir.mkdir(parents=True, exist_ok=True)
    state_meta_path = state_dir / "meta.json"
    if state_meta_path.exists() and not args.resume:
        parser.error("An interrupted restoration exists here; rerun with --resume or choose another --output")
    if state_meta_path.exists():
        state_meta = json.loads(state_meta_path.read_text(encoding="utf-8"))
        if state_meta.get("dataset") != args.dataset or Path(state_meta.get("parquet_root", "")).resolve() != source_root:
            parser.error("Existing restore state belongs to a different Parquet dataset")
    elif any(path for path in output.iterdir() if path.name != STATE_DIRNAME) and not args.resume:
        parser.error("Output directory is not empty; choose a new directory or use --resume")
    else:
        state_meta_path.write_text(json.dumps({"dataset": args.dataset, "parquet_root": str(source_root)}, indent=2) + "\n", encoding="utf-8")

    source_layout = manifest.get("class_root_relative")
    if source_layout is None:
        default_layout = "PHD08" if args.dataset == "phd08" else manifest.get("source_layout", "")
        source_layout = default_layout if default_layout in ("PHD08", "data", "retrieve_mtr_per_class") else ""
    class_root = output / source_layout if source_layout else output
    class_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "kor_synthetic":
        alphabet_file = manifest.get("alphabet_file", "alphabet.json")
        alphabet_source = source_root / alphabet_file
        if not alphabet_source.is_file():
            raise FileNotFoundError(f"Parquet dataset is missing its alphabet file: {alphabet_source}")
        source_alphabet_name = manifest.get("source_alphabet_filename", "alphabet.json")
        target_alphabet = output / source_alphabet_name
        if not target_alphabet.exists():
            shutil.copy2(alphabet_source, target_alphabet)

    completed_path = state_dir / "completed.jsonl"
    completed = set()
    if completed_path.exists():
        for line in completed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(int(json.loads(line)["class_index"]))
    tasks = []
    total_classes = len(class_files)
    for position, class_index in enumerate(sorted(class_files, key=lambda value: int(value)), start=1):
        class_index = int(class_index)
        if class_index in completed:
            continue
        relative_files = class_files[str(class_index)]
        tasks.append((args.dataset, class_index, source_root, class_root, relative_files, args.resume))
    restored_classes = len(completed)

    def record(result: tuple[int, int]) -> None:
        nonlocal restored_classes
        class_index, restored_count = result
        with completed_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"class_index": class_index, "restored_samples": restored_count}) + "\n")
        restored_classes += 1
        if restored_classes % 100 == 0 or restored_classes == total_classes:
            print(f"Restored {restored_classes}/{total_classes} classes", flush=True)

    if args.workers == 1:
        for task in tasks:
            record(_restore_class(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {}
            task_iter = iter(tasks)
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < args.workers * 2:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[executor.submit(_restore_class, task)] = task[1]
                if not pending:
                    continue
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    record(future.result())

    shutil.rmtree(state_dir)
    print(f"Restored {args.dataset} into {output}")


if __name__ == "__main__":
    main()
