"""Convert local PHD08 and KorSynthetic sources into Hub-friendly Parquet shards.

Output is partitioned by original class index so the experiment loader can read
one class at a time without scanning every Parquet shard. KorSynthetic's raw
float32 background/mask arrays are preserved; no rendered image is substituted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phd08_protocol import phd08_family_id


FORMAT = "siamese-nets-parquet-v1"
DEFAULT_ROWS_PER_SHARD = 5_000
DEFAULT_ROW_GROUP_SIZE = 100
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
STATE_DIRNAME = ".conversion_state"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels_hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _alphabet(root: Path) -> tuple[list[str], Path]:
    candidates = [root / "kor_101_mfcr.json", root / "alphabet.json", root.parent / "kor_101_mfcr.json", root.parent / "alphabet.json"]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot find kor_101_mfcr.json or alphabet.json under {root}")
    return _read_alphabet_file(path)


def _read_alphabet_file(path: Path) -> tuple[list[str], Path]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("alphabet")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], list):
        raise ValueError(f"Unexpected alphabet schema in {path}")
    symbols = []
    for item in entries[0]:
        if not isinstance(item, list) or not item or not isinstance(item[0], str):
            raise ValueError(f"Unexpected alphabet entry in {path}")
        symbols.append(item[0])
    return symbols, path


def _find_kor_class_root(root: Path) -> Path:
    for candidate in (root / "retrieve_mtr_per_class", root / "data", root):
        if candidate.is_dir() and any(path.is_dir() and path.name.isdigit() for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"Cannot find numeric class folders under {root}")


def _find_phd_root(root: Path) -> Path:
    candidate = root / "PHD08" if (root / "PHD08").is_dir() else root
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return candidate


def _class_dirs(root: Path) -> list[Path]:
    paths = [item for item in root.iterdir() if item.is_dir() and item.name.isdigit()]
    return sorted(paths, key=lambda item: int(item.name))


def _select_class_dirs(paths: list[Path], selected: set[int] | None) -> list[Path]:
    if selected is None:
        return paths
    available = {int(path.name) for path in paths}
    missing = selected - available
    if missing:
        raise ValueError(f"Requested class indices are absent: {sorted(missing)[:10]}")
    return [path for path in paths if int(path.name) in selected]


def _bucket_path(output: Path, class_index: int) -> Path:
    return output / "data" / f"{class_index // 1000:02d}"


def _pyarrow_schema(dataset: str, preview_images: bool = False) -> pa.Schema:
    common = [
        pa.field("class_index", pa.int32(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("filename", pa.string(), nullable=False),
    ]
    if dataset == "phd08":
        return pa.schema(common + [
            pa.field("family_id", pa.string(), nullable=False),
            pa.field("variant_code_1", pa.int8(), nullable=False),
            pa.field("variant_code_2", pa.int8(), nullable=False),
            pa.field("image", pa.struct([
                pa.field("bytes", pa.binary(), nullable=False),
                pa.field("path", pa.string(), nullable=False),
            ]), nullable=False),
        ])
    array = pa.list_(pa.float32(), 37 * 37)
    kor_fields = [
        pa.field("background", array, nullable=False),
        pa.field("mask", array, nullable=False),
        pa.field("shape", pa.list_(pa.int16(), 3), nullable=False),
        pa.field("background_record", pa.string(), nullable=False),
        pa.field("mask_record", pa.string(), nullable=False),
        pa.field("markup_source", pa.float32(), nullable=False),
        pa.field("symbol_index", pa.int32(), nullable=False),
    ]
    if preview_images:
        kor_fields.append(pa.field("preview_image", pa.struct([
            pa.field("bytes", pa.binary(), nullable=False),
            pa.field("path", pa.string(), nullable=False),
        ]), nullable=False))
    return pa.schema(common + kor_fields)


def _kor_row(class_index: int, symbol: str, path: Path, preview_images: bool) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) < 4:
        raise ValueError(f"KorSynthetic sample has fewer than four records: {path}")
    background, mask, markup, symbol_record = payload[:4]
    for item, expected_label in ((background, "main_data"), (mask, "main_data_mask")):
        if item.get("label") != expected_label or item.get("shape") != [37, 37, 1] or len(item.get("data", [])) != 1369:
            raise ValueError(f"Unexpected KorSynthetic image record in {path}")
    if symbol_record.get("label") != "symbols_0" or len(symbol_record.get("data", [])) != 1:
        raise ValueError(f"Unexpected KorSynthetic symbol record in {path}")
    symbol_index = int(symbol_record["data"][0])
    if symbol_index != class_index:
        raise ValueError(f"Sample label index {symbol_index} does not match folder {class_index}: {path}")
    if len(markup.get("data", [])) != 1:
        raise ValueError(f"Unexpected KorSynthetic metadata record in {path}")
    markup_value = float(markup["data"][0])
    row = {
        "class_index": class_index,
        "symbol": symbol,
        "filename": path.name,
        "background": [float(value) for value in background["data"]],
        "mask": [float(value) for value in mask["data"]],
        "shape": [int(value) for value in background["shape"]],
        "background_record": background["label"],
        "mask_record": mask["label"],
        "markup_source": markup_value,
        "symbol_index": symbol_index,
    }
    if preview_images:
        pixels = [
            max(0, min(255, round(float(bg) * float(mask_value) * 255)))
            for bg, mask_value in zip(background["data"], mask["data"])
        ]
        image = Image.new("L", (37, 37))
        image.putdata(pixels)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        row["preview_image"] = {"bytes": encoded.getvalue(), "path": path.name + ".preview.png"}
    return row


def _kor_hf_features(preview_images: bool) -> str:
    preview = '''  - name: preview_image
    dtype: image
''' if preview_images else ""
    return f'''dataset_info:
  features:
  - name: class_index
    dtype: int32
  - name: symbol
    dtype: string
  - name: filename
    dtype: string
  - name: background
    list: float32
    length: 1369
  - name: mask
    list: float32
    length: 1369
  - name: shape
    list: int16
    length: 3
  - name: background_record
    dtype: string
  - name: mask_record
    dtype: string
  - name: markup_source
    dtype: float32
  - name: symbol_index
    dtype: int32
{preview}'''


def _phd_row(class_index: int, symbol: str, path: Path) -> dict:
    family = phd08_family_id(str(class_index), path.name)
    fields = path.stem.split("_")
    return {
        "class_index": class_index,
        "symbol": symbol,
        "filename": path.name,
        "family_id": family,
        "variant_code_1": int(fields[4]),
        "variant_code_2": int(fields[5]),
        "image": {"bytes": path.read_bytes(), "path": path.name},
    }


def _source_files(dataset: str, class_dir: Path) -> list[Path]:
    if dataset == "phd08":
        suffixes = {".png"}
    else:
        suffixes = {".json"}
    return sorted(
        (entry for entry in class_dir.iterdir() if entry.is_file() and entry.suffix.lower() in suffixes),
        key=lambda path: path.name,
    )


def _class_output_files(output: Path, class_index: int) -> list[Path]:
    bucket = _bucket_path(output, class_index)
    prefix = f"class-{class_index:05d}-part-"
    return sorted(bucket.glob(prefix + "*.parquet")) if bucket.exists() else []


def _convert_class(
    dataset: str,
    class_dir: Path,
    symbol: str,
    output: Path,
    rows_per_shard: int,
    row_group_size: int,
    compression: str,
    resume: bool,
    preview_images: bool,
) -> tuple[list[str], int]:
    class_index = int(class_dir.name)
    files = _source_files(dataset, class_dir)
    if not files:
        raise ValueError(f"Class folder contains no supported samples: {class_dir}")
    if dataset == "phd08":
        if len(files) != 2187:
            raise ValueError(f"Expected 2,187 PHD08 PNG files in {class_dir}, found {len(files)}")
        variants: dict[str, set[tuple[str, str]]] = {}
        for path in files:
            family_id = phd08_family_id(str(class_index), path.name)
            fields = path.stem.split("_")
            variants.setdefault(family_id, set()).add((fields[4], fields[5]))
        expected = {(a, b) for a in "012" for b in "012"}
        if len(variants) != 243 or any(values != expected for values in variants.values()):
            raise ValueError(f"PHD08 family audit failed for class folder {class_index}")
    bucket = _bucket_path(output, class_index)
    bucket.mkdir(parents=True, exist_ok=True)
    existing = {path.name: path for path in _class_output_files(output, class_index)}
    written = []
    schema = _pyarrow_schema(dataset, preview_images)
    for part, start in enumerate(range(0, len(files), rows_per_shard)):
        if len(files) > rows_per_shard * 10 and part % 100 == 0:
            print(f"  class {class_index}: shard {part + 1}/{(len(files) + rows_per_shard - 1) // rows_per_shard}", flush=True)
        part_files = files[start:start + rows_per_shard]
        final_path = bucket / f"class-{class_index:05d}-part-{part:05d}.parquet"
        if final_path.name in existing:
            if not resume:
                raise FileExistsError(f"Output shard exists; use --resume or choose a new directory: {final_path}")
            parquet_file = pq.ParquetFile(final_path)
            if parquet_file.metadata.num_rows != len(part_files) or parquet_file.schema_arrow != schema:
                raise ValueError(f"Existing Parquet shard does not match source data: {final_path}")
            written.append(str(final_path.relative_to(output)))
            continue
        rows = [
            _phd_row(class_index, symbol, path) if dataset == "phd08" else _kor_row(class_index, symbol, path, preview_images)
            for path in part_files
        ]
        table = pa.Table.from_pylist(rows, schema=schema)
        temporary = final_path.with_suffix(final_path.suffix + ".tmp")
        pq.write_table(
            table,
            temporary,
            compression=None if compression.lower() == "none" else compression,
            row_group_size=row_group_size,
            use_dictionary=["symbol", "family_id"] if dataset == "phd08" else ["symbol", "background_record", "mask_record"],
        )
        os.replace(temporary, final_path)
        written.append(str(final_path.relative_to(output)))
    unexpected = set(existing) - {Path(item).name for item in written}
    if unexpected:
        raise ValueError(f"Unexpected stale shards for class {class_index}: {sorted(unexpected)[:3]}")
    return written, len(files)


def _convert_class_worker(task: tuple) -> tuple[int, list[str], int]:
    dataset, class_dir, symbol, output, rows_per_shard, row_group_size, compression, resume, preview_images = task
    class_dir = Path(class_dir)
    files, row_count = _convert_class(
        dataset, class_dir, symbol, Path(output), rows_per_shard,
        row_group_size, compression, resume, preview_images,
    )
    return int(class_dir.name), files, row_count


def _write_readme(output: Path, dataset: str, preview_images: bool = False) -> None:
    config_name = "PHD08" if dataset == "phd08" else "KorSynthetic"
    if dataset == "phd08":
        description = '''Each row stores one source PNG. `class_index` is the original numeric folder name; `symbol` is its position in the supplied KorSynthetic alphabet. `family_id` follows the repository rule: class folder plus the first four underscore-separated filename fields. The final two ternary filename fields are preserved as `variant_code_1` and `variant_code_2` without assigning them undocumented meanings.'''
        features = '''dataset_info:
  features:
  - name: class_index
    dtype: int32
  - name: symbol
    dtype: string
  - name: filename
    dtype: string
  - name: family_id
    dtype: string
  - name: variant_code_1
    dtype: int8
  - name: variant_code_2
    dtype: int8
  - name: image
    dtype: image
'''
    else:
        description = '''Each row stores the original 37×37 float32 background and mask arrays from one KorSynthetic JSON sample, plus original class index, Hangul symbol, sample filename and remaining source metadata. Training code samples a background and multiplies it by the mask; the arrays are kept separate to preserve that behavior.'''
        if preview_images:
            description += " The optional `preview_image` is a static visualization using this sample's own background; it is not the randomized training composite."
        features = _kor_hf_features(preview_images)
    readme = f'''---
configs:
- config_name: {config_name}
  data_files:
  - split: train
    path: "data/**/*.parquet"
{features.rstrip()}
---

# {config_name}

Converted from the local source files to Parquet by `scripts/convert_datasets_to_parquet.py`.
`dataset_manifest.json` records original class indices, class-to-shard paths and class/alphabet fingerprints.
The Hub `train` split contains the complete source collection for loading convenience; it is not an experiment split.
Experiment train/validation/test splits are defined separately by the repository manifests.

{description}

Before publishing, replace this note with the verified source citation and redistribution license.
'''
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert local PHD08/KorSynthetic data to Parquet shards")
    parser.add_argument("--dataset", choices=("phd08", "kor_synthetic"), required=True)
    parser.add_argument("--root", type=Path, required=True, help="Source root; PHD08 may contain a nested PHD08 folder")
    parser.add_argument("--output", type=Path, required=True, help="New or previously converted dataset directory")
    parser.add_argument("--alphabet-json", type=Path, help="Indexed alphabet JSON; required for PHD08 label mapping")
    parser.add_argument("--class-ids", help="Optional comma-separated subset for conversion smoke checks")
    parser.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel class converters; keep modest on shared/network storage")
    parser.add_argument("--preview-images", action="store_true", help="Add KorSynthetic Image previews to Parquet for visual inspection in the Hub viewer")
    parser.add_argument("--compression", choices=("zstd", "snappy", "none"), default="zstd")
    parser.add_argument("--resume", action="store_true", help="Resume matching completed shards in an output directory")
    args = parser.parse_args()
    if args.rows_per_shard < 1 or args.row_group_size < 1 or args.workers < 1:
        parser.error("--rows-per-shard, --row-group-size, and --workers must be positive")
    if args.dataset == "phd08" and args.alphabet_json is None:
        parser.error("--alphabet-json is required for PHD08 so numeric class folders are stored with their Hangul labels")
    if args.dataset == "phd08" and args.preview_images:
        parser.error("--preview-images is only for KorSynthetic; PHD08 already stores its PNG images")
    source_root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source_root == output or source_root in output.parents:
        parser.error("--output must not be inside the source directory")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / STATE_DIRNAME / "state.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    selected = {int(value) for value in args.class_ids.split(",")} if args.class_ids else None
    state_meta_path = output / STATE_DIRNAME / "meta.json"
    if state_meta_path.exists() and not args.resume:
        parser.error("An interrupted conversion exists here; rerun with --resume or choose another --output")
    if state_meta_path.exists():
        state_meta = json.loads(state_meta_path.read_text(encoding="utf-8"))
        if (state_meta.get("dataset") != args.dataset or
                Path(state_meta.get("source_root", "")).resolve() != source_root or
                state_meta.get("selected_class_ids") != (sorted(selected) if selected is not None else None) or
                state_meta.get("rows_per_shard") != args.rows_per_shard or
                state_meta.get("row_group_size") != args.row_group_size or
                state_meta.get("compression") != args.compression or
                state_meta.get("preview_images") != args.preview_images):
            parser.error("Existing conversion state belongs to a different source; choose another --output")
    elif any(path for path in output.iterdir() if path.name != STATE_DIRNAME) and not args.resume:
        parser.error("Output directory is not empty; choose a new directory or use --resume")
    else:
        state_meta_path.write_text(json.dumps({
            "dataset": args.dataset,
            "source_root": str(source_root),
            "selected_class_ids": sorted(selected) if selected is not None else None,
            "rows_per_shard": args.rows_per_shard,
            "row_group_size": args.row_group_size,
            "compression": args.compression,
            "preview_images": args.preview_images,
        }, indent=2) + "\n", encoding="utf-8")

    alphabet = None
    alphabet_path = None
    if args.dataset == "kor_synthetic":
        class_root = _find_kor_class_root(source_root)
        alphabet, alphabet_path = _read_alphabet_file(args.alphabet_json) if args.alphabet_json else _alphabet(source_root)
    else:
        class_root = _find_phd_root(source_root)
        alphabet, alphabet_path = _read_alphabet_file(args.alphabet_json)
    state_meta = json.loads(state_meta_path.read_text(encoding="utf-8"))
    alphabet_hash = _sha256(alphabet_path)
    saved_alphabet_hash = state_meta.get("alphabet_sha256")
    if saved_alphabet_hash is not None and saved_alphabet_hash != alphabet_hash:
        parser.error("Alphabet JSON changed during conversion; choose another --output")
    state_meta["alphabet_sha256"] = alphabet_hash
    state_meta_path.write_text(json.dumps(state_meta, indent=2) + "\n", encoding="utf-8")

    class_dirs = _select_class_dirs(_class_dirs(class_root), selected)
    if args.dataset == "phd08" and selected is None and len(class_dirs) != 2350:
        raise ValueError(f"Expected 2350 PHD08 class folders, found {len(class_dirs)}")
    if args.dataset == "phd08" and alphabet is not None:
        for class_dir in class_dirs:
            class_index = int(class_dir.name)
            if class_index >= len(alphabet) or not (0xAC00 <= ord(alphabet[class_index]) <= 0xD7A3):
                raise ValueError(f"PHD08 folder {class_index} does not map to a Hangul syllable")
    if args.dataset == "kor_synthetic" and selected is None and len(class_dirs) != len(alphabet):
        raise ValueError(f"Expected {len(alphabet)} KorSynthetic class folders, found {len(class_dirs)}")
    if args.dataset == "kor_synthetic" and selected is None:
        observed_indices = [int(path.name) for path in class_dirs]
        if observed_indices != list(range(len(alphabet))):
            raise ValueError("KorSynthetic class folders must cover all alphabet indices without gaps")
    existing_manifest_path = output / "dataset_manifest.json"
    if args.resume and existing_manifest_path.exists():
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        existing_ids = sorted(int(value) for value in existing_manifest.get("class_indices", []))
        requested_ids = sorted(int(path.name) for path in class_dirs)
        if (existing_manifest.get("dataset") != args.dataset or existing_ids != requested_ids or
                existing_manifest.get("rows_per_shard_limit") != args.rows_per_shard or
                existing_manifest.get("row_group_size") != args.row_group_size or
                existing_manifest.get("compression") != args.compression or
                existing_manifest.get("preview_images", False) != args.preview_images):
            parser.error("Existing Parquet output has a different dataset/class subset; choose another --output")

    completed_path = output / STATE_DIRNAME / "state.jsonl"
    completed = set()
    if completed_path.exists():
        for line in completed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(int(json.loads(line)["class_index"]))
    class_files: dict[str, list[str]] = {}
    class_sample_counts: dict[str, int] = {}
    tasks = []
    completed_count = 0
    for class_dir in class_dirs:
        class_index = int(class_dir.name)
        if class_index in completed:
            existing = _class_output_files(output, class_index)
            if not existing:
                raise ValueError(f"Conversion state marks class {class_index} complete but no shards exist")
            class_files[str(class_index)] = [str(path.relative_to(output)) for path in existing]
            class_sample_counts[str(class_index)] = sum(pq.ParquetFile(path).metadata.num_rows for path in existing)
            completed_count += 1
            continue
        symbol = alphabet[class_index] if alphabet is not None else str(class_index)
        tasks.append((
            args.dataset, class_dir, symbol, output, args.rows_per_shard,
            args.row_group_size, args.compression, args.resume, args.preview_images,
        ))

    if args.workers == 1:
        results = map(_convert_class_worker, tasks)
        for class_index, files, row_count in results:
            class_files[str(class_index)] = files
            class_sample_counts[str(class_index)] = row_count
            with completed_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"class_index": class_index, "parquet_files": files}, ensure_ascii=False) + "\n")
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == len(class_dirs):
                print(f"Converted {completed_count}/{len(class_dirs)} classes", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {}
            task_iter = iter(tasks)
            max_pending = max(args.workers, args.workers * 2)
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < max_pending:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    class_dir = task[1]
                    class_index = int(class_dir.name)
                    if len(pending) == 0 and (completed_count == 0 or completed_count % 100 == 0):
                        print(f"Submitting class folder {class_index} ({completed_count}/{len(class_dirs)} complete)", flush=True)
                    pending[executor.submit(_convert_class_worker, task)] = class_index
                if not pending:
                    continue
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    class_index, files, row_count = future.result()
                    class_files[str(class_index)] = files
                    class_sample_counts[str(class_index)] = row_count
                    with completed_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"class_index": class_index, "parquet_files": files}, ensure_ascii=False) + "\n")
                    completed_count += 1
                    if completed_count % 100 == 0 or completed_count == len(class_dirs):
                        print(f"Converted {completed_count}/{len(class_dirs)} classes", flush=True)

    ordered_indices = sorted((int(value) for value in class_files))
    class_names_hash = _labels_hash([str(value) for value in ordered_indices])
    class_files = {str(index): class_files[str(index)] for index in ordered_indices}
    class_sample_counts = {str(index): class_sample_counts[str(index)] for index in ordered_indices}
    manifest = {
        "format": FORMAT,
        "dataset": args.dataset,
        "schema_version": 1,
        "source_layout": str(class_root.name),
        "class_indices": ordered_indices,
        "class_files": class_files,
        "samples_per_class": class_sample_counts,
        "total_samples": sum(class_sample_counts.values()),
        "class_folder_names_sha256": class_names_hash,
        "partial_dataset": selected is not None,
        "rows_per_shard_limit": args.rows_per_shard,
        "row_group_size": args.row_group_size,
        "compression": args.compression,
        "preview_images": args.preview_images,
    }
    if args.dataset == "phd08":
        manifest["kor_synthetic_alignment"] = {
            "method": "PHD08 folder index -> same-position symbol in the explicitly supplied alphabet JSON",
            "alphabet_json_sha256": _sha256(alphabet_path),
            "aligned_classes": len(ordered_indices),
        }
    if alphabet_path is not None:
        manifest["alphabet_file"] = "alphabet.json"
        manifest["alphabet_sha256"] = alphabet_hash
        target_alphabet = output / "alphabet.json"
        if alphabet_path.resolve() != target_alphabet.resolve():
            shutil.copy2(alphabet_path, target_alphabet)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_readme(output, args.dataset, args.preview_images)
    shutil.rmtree(output / STATE_DIRNAME, ignore_errors=True)
    print(f"Wrote {manifest['total_samples']} rows across {len(class_files)} classes to {output}")


if __name__ == "__main__":
    main()
