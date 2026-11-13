"""Build deterministic class-split manifests from the local datasets.

The manifests contain class names and split rules, not copied image files.
Large PHD08/KorSynthetic trees are therefore not loaded into memory here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
DEFAULT_SEED = 20261113


def _labels_sha256(labels: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(labels)).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output: Path, manifest: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stable_split(items: Iterable[str], counts: tuple[int, ...], seed: int) -> list[list[str]]:
    values = sorted(items)
    expected = sum(counts)
    if len(values) != expected:
        raise ValueError(f"Need {expected} items for split {counts}; found {len(values)}")
    random.Random(seed).shuffle(values)
    result, offset = [], 0
    for count in counts:
        result.append(sorted(values[offset:offset + count]))
        offset += count
    return result


def _cub_root(root: Path) -> Path:
    candidates = (root, root / "CUB_200_2011")
    for candidate in candidates:
        if (candidate / "classes.txt").exists() and (candidate / "images.txt").exists():
            return candidate
    raise FileNotFoundError(f"Cannot find CUB_200_2011 metadata under {root}")


def _read_cub_classes(cub_root: Path) -> list[str]:
    rows = []
    for line in (cub_root / "classes.txt").read_text(encoding="utf-8").splitlines():
        class_id, name = line.split(maxsplit=1)
        rows.append((int(class_id), name.strip()))
    rows.sort()
    return [name for _, name in rows]


def _cub_manifest(root: Path, seed: int) -> dict:
    cub_root = _cub_root(root)
    classes = _read_cub_classes(cub_root)
    if len(classes) != 200:
        raise ValueError(f"CUB should contain 200 classes, found {len(classes)}")

    # This is the common first-100/last-100 DML class protocol. Twenty of the
    # 100 training classes are reserved for class-disjoint validation; the
    # final fit may use all 100 training classes after model selection.
    train_pool, test_classes = classes[:100], classes[100:]
    train_classes, valid_classes = _stable_split(train_pool, (80, 20), seed)

    image_rows = (cub_root / "images.txt").read_text(encoding="utf-8").splitlines()
    label_rows = (cub_root / "image_class_labels.txt").read_text(encoding="utf-8").splitlines()
    split_rows = (cub_root / "train_test_split.txt").read_text(encoding="utf-8").splitlines()
    hashes = {
        name: _sha256(cub_root / name)
        for name in ("classes.txt", "images.txt", "image_class_labels.txt", "train_test_split.txt")
    }
    image_root = cub_root / "images"
    listed = {line.split(maxsplit=1)[1].strip() for line in image_rows}
    present = {str(path.relative_to(image_root)).replace("\\", "/") for path in image_root.rglob("*.jpg")}
    extras = sorted(present - listed)
    missing = sorted(listed - present)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "cub200",
        "seed": seed,
        "source": "CUB-200-2011 metadata files; image files are referenced by images.txt",
        "counts": {"classes": len(classes), "metadata_images": len(image_rows), "present_jpg": len(present)},
        "catalog_labels_sha256": _labels_sha256(classes),
        "metadata_sha256": hashes,
        "protocols": {
            "closed_set": {
                "classes": classes,
                "image_split": "official train_test_split.txt; reserve validation by stable hash over official train images",
            },
            "class_disjoint_retrieval": {
                "development_train_classes": train_classes,
                "development_validation_classes": valid_classes,
                "final_train_classes": train_pool,
                "test_classes": test_classes,
                "class_protocol": "first 100 classes for train/validation, last 100 classes for test",
            },
        },
        "integrity": {"unlisted_jpgs": extras, "missing_metadata_images": missing},
    }


def _clinc_manifest(root: Path, seed: int) -> dict:
    candidates = [root] if root.is_file() else [root / "data_full.json", root / "clinc150.json"]
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(f"Need data_full.json or clinc150.json under {root}")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    required = ("train", "val", "test", "oos_train", "oos_val", "oos_test")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"CLINC file is missing splits: {missing}")
    labels = sorted({str(row[1]) for row in payload["train"] if row[1] != "oos"})
    if len(labels) != 150:
        raise ValueError(f"Expected 150 in-scope intents, found {len(labels)}")
    train_classes, valid_classes, test_classes = _stable_split(labels, (90, 30, 30), seed)
    split_counts = {key: len(payload[key]) for key in required}
    hashes = {data_path.name: _sha256(data_path)}
    def normalize_text(text: str) -> str:
        return " ".join(str(text).casefold().split())

    normalized = {
        split: [(normalize_text(row[0]), str(row[1])) for row in payload[split]]
        for split in ("train", "val", "test")
    }
    overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_labels = {}
        right_labels = {}
        for text, label in normalized[left]:
            left_labels.setdefault(text, set()).add(label)
        for text, label in normalized[right]:
            right_labels.setdefault(text, set()).add(label)
        for text in sorted(left_labels.keys() & right_labels.keys()):
            overlaps.append({
                "left_split": left,
                "right_split": right,
                "text": text,
                "left_labels": sorted(left_labels[text]),
                "right_labels": sorted(right_labels[text]),
                "label_conflict": left_labels[text] != right_labels[text],
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "clinc150",
        "seed": seed,
        "source_file": data_path.name,
        "source_sha256": hashes[data_path.name],
        "counts": split_counts,
        "intents": labels,
        "catalog_labels_sha256": _labels_sha256(labels + ["oos"]),
        "protocols": {
            "closed_set": {"classes": labels, "source_splits": {"train": "train", "valid": "val", "test": "test"}},
            "class_disjoint_retrieval": {
                "train_classes": train_classes,
                "validation_classes": valid_classes,
                "test_classes": test_classes,
                "final_train_classes": sorted(train_classes + valid_classes),
                "gallery_split": "train",
                "validation_query_split": "val",
                "test_query_split": "test",
                "oos_protocol": {"calibration_split": "oos_val", "test_split": "oos_test"},
            },
        },
        "integrity": {
            "duplicate_texts_are_preserved": True,
            "normalized_cross_split_overlaps": overlaps,
            "intra_split_duplicate_rows": {
                split: len(rows) - len({text for text, _ in rows}) for split, rows in normalized.items()
            },
        },
    }


def _omniglot_root(root: Path) -> Path:
    for candidate in (root, root / "python"):
        if (candidate / "images_background").is_dir() and (candidate / "images_evaluation").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find Omniglot python/images_* under {root}")


def _omniglot_manifest(root: Path, seed: int) -> dict:
    data_root = _omniglot_root(root)
    background = sorted(path.name for path in (data_root / "images_background").iterdir() if path.is_dir())
    evaluation = sorted(path.name for path in (data_root / "images_evaluation").iterdir() if path.is_dir())
    if (len(background), len(evaluation)) != (30, 20):
        raise ValueError(f"Expected 30 background and 20 evaluation alphabets; got {len(background)} and {len(evaluation)}")
    train_alphabets, valid_alphabets = _stable_split(background, (25, 5), seed)

    def class_names(split: str, alphabets: list[str]) -> list[str]:
        names = []
        for alphabet in alphabets:
            path = data_root / ("images_background" if split == "background" else "images_evaluation") / alphabet
            names.extend(f"{alphabet}/{item.name}" for item in path.iterdir() if item.is_dir())
        return sorted(names)

    background_classes = class_names("background", background)
    evaluation_classes = class_names("evaluation", evaluation)
    train_classes = class_names("background", train_alphabets)
    validation_classes = class_names("background", valid_alphabets)
    image_count_distribution: dict[str, int] = {}
    for split, alphabets in (("images_background", background), ("images_evaluation", evaluation)):
        split_root = data_root / split
        for alphabet in alphabets:
            for character in (split_root / alphabet).iterdir():
                if character.is_dir():
                    count = sum(1 for image in character.iterdir() if image.is_file())
                    image_count_distribution[str(count)] = image_count_distribution.get(str(count), 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "omniglot",
        "seed": seed,
        "counts": {"background_alphabets": len(background), "evaluation_alphabets": len(evaluation), "background": len(background_classes), "evaluation": len(evaluation_classes), "images_per_class": image_count_distribution},
        "catalog_labels_sha256": _labels_sha256(background_classes + evaluation_classes),
        "protocols": {
            "closed_set": {
                "classes": background_classes + evaluation_classes,
                "alphabets": background + evaluation,
                "sample_split": "stable_hash(uid, seed) into 12/4/4 for class-level episode datasets",
            },
            "class_disjoint_retrieval": {
                "development_train_alphabets": train_alphabets,
                "development_validation_alphabets": valid_alphabets,
                "final_train_alphabets": background,
                "test_alphabets": evaluation,
                "development_train_classes": train_classes,
                "development_validation_classes": validation_classes,
                "final_train_classes": background_classes,
                "test_classes": evaluation_classes,
                "class_key": "alphabet/character",
            },
        },
    }


def _phd_root(root: Path) -> Path:
    candidate = root / "PHD08" if (root / "PHD08").is_dir() else root
    if not candidate.is_dir():
        raise FileNotFoundError(root)
    return candidate


def _phd_manifest(root: Path, seed: int, deep_scan: bool = False) -> dict:
    data_root = _phd_root(root)
    labels = sorted((path.name for path in data_root.iterdir() if path.is_dir()), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    if len(labels) != 2350:
        raise ValueError(f"Expected 2350 PHD08 class folders, found {len(labels)}")
    train_classes, valid_classes, test_classes = _stable_split(labels, (1504, 376, 470), seed)
    integrity = {"full_image_scan_performed": False, "source_family_metadata": "unknown"}
    counts = {"classes": len(labels), "expected_images_per_class": 2187, "expected_total_images": 5139450}
    if deep_scan:
        distribution: dict[str, int] = {}
        observed_total = 0
        for label in labels:
            class_dir = data_root / label
            count = sum(1 for item in class_dir.iterdir() if item.is_file() and item.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
            distribution[str(count)] = distribution.get(str(count), 0) + 1
            observed_total += count
        integrity["full_image_scan_performed"] = True
        integrity["images_per_class_distribution"] = distribution
        counts["observed_total_images"] = observed_total
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "phd08",
        "seed": seed,
        "counts": counts,
        "catalog_labels_sha256": _labels_sha256(labels),
        "class_names_sha256": hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest(),
        "protocols": {
            "closed_set": {"classes": labels, "sample_split": "stable_hash(uid, seed) into 60/20/20; not source-disjoint"},
            "class_disjoint_retrieval": {
                "train_classes": train_classes,
                "validation_classes": valid_classes,
                "test_classes": test_classes,
                "final_train_classes": sorted(train_classes + valid_classes),
                "source_disjoint": False,
                "note": "Source/family metadata must be supplied separately before claiming source-disjoint evaluation.",
            },
        },
        "integrity": integrity,
    }


def _kor_manifest(root: Path, seed: int) -> dict:
    base = root / "retrieve_mtr_per_class" if (root / "retrieve_mtr_per_class").is_dir() else root
    alphabet_path = root / "kor_101_mfcr.json" if (root / "kor_101_mfcr.json").exists() else base.parent / "kor_101_mfcr.json"
    if not alphabet_path.exists():
        raise FileNotFoundError(f"Cannot find kor_101_mfcr.json adjacent to {root}")
    alphabet = [entry[0] for entry in json.loads(alphabet_path.read_text(encoding="utf-8"))["alphabet"][0]]
    class_dirs = sorted((path.name for path in base.iterdir() if path.is_dir()), key=lambda name: (not name.isdigit(), int(name) if name.isdigit() else name))
    if len(class_dirs) != len(alphabet):
        raise ValueError(f"Folder/alphabet count mismatch: {len(class_dirs)} folders, {len(alphabet)} symbols")
    hangul_ids = [index for index, symbol in enumerate(alphabet) if 0xAC00 <= ord(symbol) <= 0xD7A3]
    hangul_id_set = set(hangul_ids)
    other_ids = [index for index in range(len(alphabet)) if index not in hangul_id_set]
    hangul = [alphabet[index] for index in hangul_ids]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "kor_synthetic",
        "seed": seed,
        "counts": {"all_symbols": len(alphabet), "hangul_syllables": len(hangul), "non_hangul_symbols": len(other_ids)},
        "alphabet_sha256": _sha256(alphabet_path),
        "class_folder_names_sha256": hashlib.sha256("\n".join(class_dirs).encode("utf-8")).hexdigest(),
        "class_folder_to_symbol": {class_dirs[index]: alphabet[index] for index in range(len(class_dirs))},
        "catalog_labels_sha256": {
            "hangul_only": _labels_sha256(hangul),
            "all_symbols": _labels_sha256(alphabet),
        },
        "protocols": {
            "hangul_only": {"classes": hangul, "excluded_folder_ids": [class_dirs[index] for index in other_ids]},
            "all_symbols": {"classes": alphabet},
        },
        "non_hangul_symbols": [alphabet[index] for index in other_ids],
    }


BUILDERS = {
    "clinc150": _clinc_manifest,
    "cub200": _cub_manifest,
    "omniglot": _omniglot_manifest,
    "phd08": _phd_manifest,
    "kor_synthetic": _kor_manifest,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic experiment split manifests")
    parser.add_argument("--dataset", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--deep-scan", action="store_true", help="Count PHD08 image files in all 2350 class folders")
    args = parser.parse_args()
    if args.dataset == "phd08":
        manifest = _phd_manifest(args.root, args.seed, deep_scan=args.deep_scan)
    else:
        manifest = BUILDERS[args.dataset](args.root, args.seed)
    _write_manifest(args.output, manifest)
    print(f"Wrote {args.dataset} manifest: {args.output}")
    print(json.dumps(manifest.get("counts", {}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
