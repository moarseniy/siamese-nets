"""Build deterministic class-split manifests from the local datasets.

The manifests contain class names and split rules, not copied image files.
Large PHD08/KorSynthetic trees are therefore not loaded into memory here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phd08_protocol import PHD08_BASE_FAMILIES_PER_CLASS, PHD08_FAMILY_VARIANTS_PER_BASE, phd08_family_id


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


def _kor_class_root(root: Path) -> Path:
    for candidate in (root / "retrieve_mtr_per_class", root / "data", root):
        if candidate.is_dir() and any(path.is_dir() and path.name.isdigit() for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"Cannot find numeric KorSynthetic class folders under {root}")


def _kor_alphabet_path(root: Path, class_root: Path) -> Path:
    candidates = [root / "kor_101_mfcr.json", root / "alphabet.json", class_root.parent / "kor_101_mfcr.json", class_root.parent / "alphabet.json"]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot find kor_101_mfcr.json or alphabet.json under {root}")
    return path


def _read_kor_alphabet(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("alphabet")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], list):
        raise ValueError(f"Unexpected KorSynthetic alphabet schema in {path}")
    return [entry[0] for entry in entries[0]]


def _phd_manifest(root: Path, seed: int, deep_scan: bool = False, kor_root: Path | None = None) -> dict:
    data_root = _phd_root(root)
    labels = sorted((path.name for path in data_root.iterdir() if path.is_dir()), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    if len(labels) != 2350:
        raise ValueError(f"Expected 2350 PHD08 class folders, found {len(labels)}")
    if any(not label.isdigit() for label in labels) or len({int(label) for label in labels}) != len(labels):
        raise ValueError("PHD08 class folders must have unique numeric names used as alphabet indices")
    train_classes, valid_classes, test_classes = _stable_split(labels, (1504, 376, 470), seed)
    integrity = {
        "full_image_scan_performed": False,
        "source_family_metadata": "filename-derived candidate; full pattern audit not performed",
    }
    counts = {"classes": len(labels), "expected_images_per_class": 2187, "expected_total_images": 5139450}
    if deep_scan:
        distribution: dict[str, int] = {}
        observed_total = 0
        family_count_distribution: dict[str, int] = {}
        family_size_distribution: dict[str, int] = {}
        family_audit_errors = []
        expected_variants = {tuple(pair) for pair in ((a, b) for a in "012" for b in "012")}
        for label in labels:
            class_dir = data_root / label
            count = 0
            family_variants: dict[str, set[tuple[str, str]]] = {}
            for item in class_dir.iterdir():
                if not item.is_file() or item.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                    continue
                count += 1
                try:
                    family_id = phd08_family_id(label, item.name)
                except ValueError:
                    family_audit_errors.append(f"{label}/{item.name}: unexpected filename")
                    continue
                fields = item.stem.split("_")
                family_variants.setdefault(family_id, set()).add((fields[4], fields[5]))
            distribution[str(count)] = distribution.get(str(count), 0) + 1
            observed_total += count
            family_count_distribution[str(len(family_variants))] = family_count_distribution.get(str(len(family_variants)), 0) + 1
            for family_id, variants in family_variants.items():
                family_size_distribution[str(len(variants))] = family_size_distribution.get(str(len(variants)), 0) + 1
                if variants != expected_variants:
                    family_audit_errors.append(f"{family_id}: expected all 9 final-index combinations")
            if len(family_variants) != PHD08_BASE_FAMILIES_PER_CLASS:
                family_audit_errors.append(f"{label}: expected {PHD08_BASE_FAMILIES_PER_CLASS} filename families, found {len(family_variants)}")
        integrity["full_image_scan_performed"] = True
        integrity["images_per_class_distribution"] = distribution
        integrity["filename_family_audit"] = {
            "rule": "class folder + first four underscore-separated filename fields",
            "groups_per_class_distribution": family_count_distribution,
            "variants_per_family_distribution": family_size_distribution,
            "expected_groups_per_class": PHD08_BASE_FAMILIES_PER_CLASS,
            "expected_variants_per_family": PHD08_FAMILY_VARIANTS_PER_BASE,
            "verified": not family_audit_errors,
            "errors": family_audit_errors[:20],
        }
        if family_audit_errors:
            raise ValueError(f"PHD08 filename-family audit failed: {family_audit_errors[:5]}")
        integrity["source_family_metadata"] = "filename-derived family pattern verified in every class"
        counts["observed_total_images"] = observed_total
    p5_protocol = None
    p5_symbols = None
    alignment_alphabet_sha256 = None
    if kor_root is not None:
        kor_root = kor_root.expanduser().resolve()
        kor_classes = _kor_class_root(kor_root)
        alphabet_path = _kor_alphabet_path(kor_root, kor_classes)
        alphabet = _read_kor_alphabet(alphabet_path)
        kor_folder_ids = {int(path.name) for path in kor_classes.iterdir() if path.is_dir() and path.name.isdigit()}
        p5_symbols = []
        for label in labels:
            if not label.isdigit() or int(label) >= len(alphabet):
                raise ValueError(f"PHD08 class folder {label!r} is not a valid KorSynthetic alphabet index")
            symbol = alphabet[int(label)]
            if not 0xAC00 <= ord(symbol) <= 0xD7A3:
                raise ValueError(f"PHD08 class folder {label} maps to non-Hangul symbol {symbol!r}")
            if int(label) not in kor_folder_ids:
                raise ValueError(f"PHD08 class folder {label} has no matching KorSynthetic folder")
            p5_symbols.append(symbol)
        p5_symbols = sorted(set(p5_symbols))
        if len(p5_symbols) != len(labels):
            raise ValueError("PHD08 indices do not map one-to-one to Hangul symbols")
        alignment_alphabet_sha256 = _sha256(alphabet_path)
        p5_protocol = {"classes": p5_symbols, "label_space": "PHD08 2350-class subset mapped through alphabet[index]"}
        integrity["kor_synthetic_alignment"] = {
            "method": "folder index -> same-position symbol in alphabet.json or kor_101_mfcr.json",
            "p5_class_count": len(p5_symbols),
            "alphabet_sha256": alignment_alphabet_sha256,
            "all_indices_present_in_kor_synthetic": True,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "phd08",
        "seed": seed,
        "counts": counts,
        "catalog_labels_sha256": _labels_sha256(labels),
        "class_names_sha256": hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest(),
        "p5_catalog_labels_sha256": _labels_sha256(p5_symbols) if p5_symbols is not None else None,
        "protocols": {
            "closed_set": {"classes": labels, "sample_split": "deterministic 60/20/20 split by filename-derived source family; not font-disjoint"},
            "class_disjoint_retrieval": {
                "train_classes": train_classes,
                "validation_classes": valid_classes,
                "test_classes": test_classes,
                "final_train_classes": sorted(train_classes + valid_classes),
                "source_disjoint": False,
                "within_class_family_split": "gallery/query use disjoint filename-derived families",
                "font_disjoint": False,
                "note": "Class split is disjoint; whole-font generalization is a separate protocol.",
            },
            **({"p5_synthetic_transfer": p5_protocol} if p5_protocol is not None else {}),
        },
        "integrity": integrity,
    }


def _kor_manifest(root: Path, seed: int, phd_root: Path | None = None) -> dict:
    base = _kor_class_root(root)
    alphabet_path = _kor_alphabet_path(root, base)
    alphabet = _read_kor_alphabet(alphabet_path)
    class_dirs = sorted((path.name for path in base.iterdir() if path.is_dir()), key=lambda name: (not name.isdigit(), int(name) if name.isdigit() else name))
    if len(class_dirs) != len(alphabet):
        raise ValueError(f"Folder/alphabet count mismatch: {len(class_dirs)} folders, {len(alphabet)} symbols")
    if any(not name.isdigit() for name in class_dirs) or len({int(name) for name in class_dirs}) != len(class_dirs):
        raise ValueError("KorSynthetic class folders must have unique numeric names used as alphabet indices")
    kor_folder_ids = {int(name) for name in class_dirs}
    if kor_folder_ids != set(range(len(alphabet))):
        raise ValueError("KorSynthetic folder indices must cover the alphabet positions exactly")
    hangul_ids = [index for index, symbol in enumerate(alphabet) if 0xAC00 <= ord(symbol) <= 0xD7A3]
    hangul_id_set = set(hangul_ids)
    other_ids = [index for index in range(len(alphabet)) if index not in hangul_id_set]
    hangul = [alphabet[index] for index in hangul_ids]
    protocols = {
        "hangul_only": {"classes": hangul, "excluded_folder_ids": [class_dirs[index] for index in other_ids]},
        "all_symbols": {"classes": alphabet},
    }
    phd_symbols = None
    if phd_root is not None:
        phd_data_root = _phd_root(phd_root.expanduser().resolve())
        phd_ids = sorted((path.name for path in phd_data_root.iterdir() if path.is_dir()), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        phd_symbols = []
        for label in phd_ids:
            if not label.isdigit() or int(label) >= len(alphabet):
                raise ValueError(f"PHD08 class folder {label!r} is not an index in the KorSynthetic alphabet")
            symbol = alphabet[int(label)]
            if not 0xAC00 <= ord(symbol) <= 0xD7A3:
                raise ValueError(f"PHD08 class folder {label} maps to non-Hangul symbol {symbol!r}")
            if int(label) not in kor_folder_ids:
                raise ValueError(f"PHD08 class folder {label} has no KorSynthetic folder")
            phd_symbols.append(symbol)
        if len(phd_symbols) != len(set(phd_symbols)):
            raise ValueError("PHD08 indices do not map one-to-one to Hangul symbols")
        phd_symbols = sorted(phd_symbols)
        protocols["phd08_2350"] = {"classes": phd_symbols, "label_space": "PHD08 folder index mapped through alphabet[index]"}
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "kor_synthetic",
        "seed": seed,
        "counts": {"all_symbols": len(alphabet), "hangul_syllables": len(hangul), "non_hangul_symbols": len(other_ids)},
        "alphabet_sha256": _sha256(alphabet_path),
        "class_folder_names_sha256": hashlib.sha256("\n".join(class_dirs).encode("utf-8")).hexdigest(),
        "catalog_labels_sha256": {
            "hangul_only": _labels_sha256(hangul),
            "all_symbols": _labels_sha256(alphabet),
        },
        "protocols": protocols,
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
    parser.add_argument("--kor-root", type=Path, help="KorSynthetic root used to verify PHD08 index-to-symbol alignment")
    parser.add_argument("--phd-root", type=Path, help="PHD08 root used to add the 2350-class transfer protocol to KorSynthetic")
    args = parser.parse_args()
    if args.dataset == "phd08":
        kor_root = args.kor_root
        if kor_root is None and (args.root.resolve().parent / "kor_synthetic").is_dir():
            kor_root = args.root.resolve().parent / "kor_synthetic"
        manifest = _phd_manifest(args.root, args.seed, deep_scan=args.deep_scan, kor_root=kor_root)
    elif args.dataset == "kor_synthetic":
        phd_root = args.phd_root
        if phd_root is None and (args.root.resolve().parent / "PHD08_dataset").is_dir():
            phd_root = args.root.resolve().parent / "PHD08_dataset"
        manifest = _kor_manifest(args.root, args.seed, phd_root=phd_root)
    else:
        manifest = BUILDERS[args.dataset](args.root, args.seed)
    _write_manifest(args.output, manifest)
    print(f"Wrote {args.dataset} manifest: {args.output}")
    print(json.dumps(manifest.get("counts", {}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
