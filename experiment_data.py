"""Dataset catalogs shared by the experiment preparer, trainer, and evaluator.

Large directory datasets (PHD08 and KorSynthetic) are indexed by class folder
and read lazily. This avoids holding millions of Python path objects in RAM.
"""

from __future__ import annotations

import json
import hashlib
import io
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from phd08_protocol import phd08_family_id, phd08_family_splits


Modality = Literal["image", "text", "kor_synthetic"]


def stable_bucket(value: str, seed: int) -> int:
    """Deterministic 0..99 bucket that does not depend on Python hash randomization."""
    import hashlib

    digest = hashlib.blake2b(f"{seed}:{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 100


@dataclass(frozen=True)
class SampleRef:
    label: str
    uid: str
    value: str
    source_split: str | None = None
    family_id: str | None = None
    parquet_path: str | None = None
    parquet_row_group: int | None = None
    parquet_row_offset: int | None = None


class ClassSource:
    """A class represented by either explicit samples or a lazily scanned folder."""

    def __init__(
        self,
        label: str,
        modality: Modality,
        records: list[SampleRef] | None = None,
        folder: Path | None = None,
        image_split_file: dict[str, str] | None = None,
        uid_prefix: str | None = None,
        family_grouping: bool = False,
        parquet_files: list[Path] | None = None,
    ):
        if records is None and folder is None and not parquet_files:
            raise ValueError("ClassSource needs records or a folder")
        self.label = label
        self.modality = modality
        self.records = records
        self.folder = folder
        self.image_split_file = image_split_file or {}
        self.uid_prefix = uid_prefix or label
        self.family_grouping = bool(family_grouping)
        self.parquet_files = tuple(Path(path) for path in (parquet_files or ()))
        self._file_cache: OrderedDict[str, tuple[SampleRef, ...]] = OrderedDict()
        self._family_cache: OrderedDict[str, dict[str, tuple[SampleRef, ...]]] = OrderedDict()
        self._family_split_cache: dict[tuple[int, str], dict[str, set[str]]] = {}
        self._records_for_cache: OrderedDict[tuple[str, int, str], tuple[SampleRef, ...]] = OrderedDict()
        self._representative_cache: OrderedDict[tuple[str, int, str, int], tuple[SampleRef, ...]] = OrderedDict()

    def _folder_records(self) -> tuple[SampleRef, ...]:
        if self.parquet_files:
            return self._parquet_records()
        key = str(self.folder)
        if key in self._file_cache:
            self._file_cache.move_to_end(key)
            return self._file_cache[key]
        assert self.folder is not None
        allowed = (".png",) if self.family_grouping else (".png", ".jpg", ".jpeg", ".bmp", ".json")
        with os.scandir(self.folder) as entries:
            names = sorted(
                entry.name for entry in entries
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(allowed)
            )
        records = tuple(
            SampleRef(
                self.label,
                f"{self.uid_prefix}/{name}",
                str(self.folder / name),
                family_id=phd08_family_id(self.uid_prefix, name) if self.family_grouping else None,
            )
            for name in names
        )
        self._file_cache[key] = records
        while len(self._file_cache) > 128:
            self._file_cache.popitem(last=False)
        return records

    def _parquet_records(self) -> tuple[SampleRef, ...]:
        cache_key = "parquet:" + "|".join(map(str, self.parquet_files))
        if cache_key in self._file_cache:
            self._file_cache.move_to_end(cache_key)
            return self._file_cache[cache_key]
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading Parquet datasets requires pyarrow; install requirements-experiments.txt") from exc
        records: list[SampleRef] = []
        for parquet_path in self.parquet_files:
            parquet_file = pq.ParquetFile(parquet_path)
            required_columns = ["filename"]
            if self.family_grouping:
                required_columns.append("family_id")
            if not set(required_columns).issubset(parquet_file.schema_arrow.names):
                raise ValueError(f"Parquet class shard is missing columns {required_columns}: {parquet_path}")
            for row_group in range(parquet_file.metadata.num_row_groups):
                metadata = parquet_file.read_row_group(row_group, columns=required_columns).to_pylist()
                for row_offset, row in enumerate(metadata):
                    filename = str(row["filename"])
                    family_id = row.get("family_id")
                    if self.family_grouping:
                        expected_family_id = phd08_family_id(self.uid_prefix, filename)
                        if family_id != expected_family_id:
                            raise ValueError(f"PHD08 Parquet family ID mismatch for {self.uid_prefix}/{filename}")
                    records.append(SampleRef(
                        self.label,
                        f"{self.uid_prefix}/{filename}",
                        filename,
                        family_id=family_id,
                        parquet_path=str(parquet_path),
                        parquet_row_group=row_group,
                        parquet_row_offset=row_offset,
                    ))
        result = tuple(records)
        self._file_cache[cache_key] = result
        while len(self._file_cache) > 128:
            self._file_cache.popitem(last=False)
        return result

    def all_records(self) -> tuple[SampleRef, ...]:
        return tuple(self.records) if self.records is not None else self._folder_records()

    def _family_groups(self) -> dict[str, tuple[SampleRef, ...]]:
        cache_key = str(self.folder) if self.folder is not None else "parquet:" + "|".join(map(str, self.parquet_files))
        if cache_key in self._family_cache:
            self._family_cache.move_to_end(cache_key)
            return self._family_cache[cache_key]
        groups: dict[str, list[SampleRef]] = defaultdict(list)
        for record in self.all_records():
            if record.family_id is None:
                raise ValueError(f"Missing family ID for source-grouped class {self.label!r}")
            groups[record.family_id].append(record)
        result = {family_id: tuple(records) for family_id, records in groups.items()}
        self._family_cache[cache_key] = result
        while len(self._family_cache) > 4:
            self._family_cache.popitem(last=False)
        return result

    def _family_ids_for_split(self, split: str, seed: int, protocol: str) -> set[str]:
        groups = self._family_groups()
        if protocol == "phd08_class_disjoint" and split == "train":
            return set(groups)
        cache_key = (int(seed), protocol)
        if cache_key not in self._family_split_cache:
            self._family_split_cache[cache_key] = phd08_family_splits(list(groups), int(seed))
        if split not in self._family_split_cache[cache_key]:
            raise ValueError(f"Unknown PHD08 family split {split!r}")
        return self._family_split_cache[cache_key][split]

    def records_for(
        self,
        split: str,
        seed: int,
        protocol: str,
    ) -> tuple[SampleRef, ...]:
        key = (split, int(seed), protocol)
        if key in self._records_for_cache:
            self._records_for_cache.move_to_end(key)
            return self._records_for_cache[key]
        records = self._records_for_uncached(split, seed, protocol)
        self._records_for_cache[key] = records
        while len(self._records_for_cache) > 4:
            self._records_for_cache.popitem(last=False)
        return records

    def _records_for_uncached(
        self,
        split: str,
        seed: int,
        protocol: str,
    ) -> tuple[SampleRef, ...]:
        records = self.all_records()
        if protocol == "clinc150":
            if split in ("gallery", "train"):
                wanted = "train"
            elif split in ("validation_query", "valid"):
                wanted = "val"
            elif split in ("test_query", "test"):
                wanted = "test"
            elif split.startswith("oos_"):
                wanted = split
            else:
                raise ValueError(f"Unknown CLINC split {split!r}")
            return tuple(record for record in records if record.source_split == wanted)

        if protocol == "cub200_class_disjoint":
            if split in ("gallery", "train"):
                return tuple(record for record in records if self.image_split_file.get(record.uid) == "train")
            if split in ("query", "test"):
                return tuple(record for record in records if self.image_split_file.get(record.uid) == "test")
            if split == "valid":
                return tuple(
                    record for record in records
                    if self.image_split_file.get(record.uid) == "train" and stable_bucket(record.uid, seed) < 20
                )

        # Closed-set image splits are deterministic by sample ID. For datasets
        # with a supplied official image split, preserve that source split and
        # reserve validation only from its training half.
        if protocol == "cub200_closed" and self.image_split_file:
            if split == "train":
                return tuple(
                    record for record in records
                    if self.image_split_file.get(record.uid) == "train" and stable_bucket(record.uid, seed) >= 20
                )
            if split == "valid":
                return tuple(
                    record for record in records
                    if self.image_split_file.get(record.uid) == "train" and stable_bucket(record.uid, seed) < 20
                )
            if split == "test":
                return tuple(record for record in records if self.image_split_file.get(record.uid) == "test")

        if protocol == "omniglot_closed":
            ordered = sorted(records, key=lambda record: (stable_bucket(record.uid, seed), record.uid))
            n_train = round(0.6 * len(ordered))
            n_valid = round(0.2 * len(ordered))
            if split == "train":
                return tuple(ordered[:n_train])
            if split == "valid":
                return tuple(ordered[n_train:n_train + n_valid])
            if split == "test":
                return tuple(ordered[n_train + n_valid:])

        if protocol == "omniglot_class_disjoint":
            if split in ("gallery", "query"):
                return records
            if split in ("support", "episode_query"):
                ordered = sorted(records, key=lambda record: (stable_bucket(record.uid, seed), record.uid))
                midpoint = len(ordered) // 2
                return tuple(ordered[:midpoint] if split == "support" else ordered[midpoint:])

        if self.family_grouping and protocol in ("phd08_closed", "phd08_class_disjoint"):
            if split == "all":
                return records
            if protocol == "phd08_class_disjoint" and split == "train":
                # Class labels define the train/validation/test split in P3/P4.
                # Use every image from training classes; split families only
                # between gallery and query within held-out classes.
                return records
            groups = self._family_groups()
            allowed_families = self._family_ids_for_split(split, seed, protocol)
            return tuple(record for family in sorted(allowed_families) for record in groups[family])

        bucket = lambda record: stable_bucket(record.uid, seed)
        if split == "train":
            return tuple(record for record in records if bucket(record) < 60)
        if split == "valid":
            return tuple(record for record in records if 60 <= bucket(record) < 80)
        if split == "test":
            return tuple(record for record in records if bucket(record) >= 80)
        if split == "gallery":
            return tuple(record for record in records if bucket(record) < 50)
        if split == "query":
            return tuple(record for record in records if bucket(record) >= 50)
        if split == "all":
            return records
        raise ValueError(f"Unknown data split {split!r} for {protocol!r}")

    def sample(self, split: str, seed: int, protocol: str, k: int, rng: random.Random, max_examples: int | None = None) -> list[SampleRef]:
        if self.family_grouping and protocol in ("phd08_closed", "phd08_class_disjoint"):
            groups = self._family_groups()
            family_ids = sorted(self._family_ids_for_split(split, seed, protocol))
            if max_examples is not None:
                representative_key = (split, int(seed), protocol, int(max_examples))
                if representative_key not in self._representative_cache:
                    selected_families = sorted(
                        family_ids,
                        key=lambda family: (stable_bucket(family, seed + 7919), family),
                    )[:max_examples]
                    representatives = tuple(
                        min(groups[family], key=lambda record: (stable_bucket(record.uid, seed + 15401), record.uid))
                        for family in selected_families
                    )
                    self._representative_cache[representative_key] = representatives
                    while len(self._representative_cache) > 4:
                        self._representative_cache.popitem(last=False)
                else:
                    self._representative_cache.move_to_end(representative_key)
                representatives = self._representative_cache[representative_key]
                if len(representatives) < k:
                    raise ValueError(f"Class {self.label!r} has {len(representatives)} source families in {split}, needs {k}")
                return rng.sample(list(representatives), k)
            if len(family_ids) < k:
                raise ValueError(f"Class {self.label!r} has {len(family_ids)} source families in {split}, needs {k}")
            selected_families = rng.sample(family_ids, k)
            return [rng.choice(groups[family]) for family in selected_families]
        eligible = self.records_for(split, seed, protocol)
        if max_examples is not None:
            eligible = self.budget_records(split, seed, protocol, max_examples)
        if len(eligible) < k:
            raise ValueError(f"Class {self.label!r} has {len(eligible)} eligible samples in {split}, needs {k}")
        return rng.sample(eligible, k)

    def budget_records(self, split: str, seed: int, protocol: str, max_examples: int | None) -> tuple[SampleRef, ...]:
        """Return a fixed, nested low-shot reference set for one class."""
        eligible = self.records_for(split, seed, protocol)
        if max_examples is None:
            return eligible
        if self.family_grouping and protocol in ("phd08_closed", "phd08_class_disjoint"):
            groups = self._family_groups()
            family_ids = sorted({record.family_id for record in eligible if record.family_id is not None})
            selected = sorted(
                family_ids,
                key=lambda family: (stable_bucket(family, seed + 7919), family),
            )[:max_examples]
            return tuple(
                min(groups[family], key=lambda record: (stable_bucket(record.uid, seed + 15401), record.uid))
                for family in selected
            )
        return tuple(sorted(eligible, key=lambda record: (stable_bucket(record.uid, seed + 7919), record.uid))[:max_examples])


class DatasetCatalog:
    def __init__(self, name: str, modality: Modality, classes: dict[str, ClassSource], fingerprints: dict[str, str] | None = None):
        self.name = name
        self.modality = modality
        self.classes = dict(sorted(classes.items()))
        self.labels = list(self.classes)
        self.fingerprints = fingerprints or {}
        if not self.labels:
            raise ValueError(f"No classes found for {name}")

    def references(self, labels: list[str], split: str, seed: int, protocol: str) -> Iterator[SampleRef]:
        for label in labels:
            yield from self.classes[label].records_for(split, seed, protocol)

    def sample_class(self, label: str, split: str, seed: int, protocol: str, k: int, rng: random.Random, max_examples: int | None = None):
        return self.classes[label].sample(split, seed, protocol, k, rng, max_examples)

    def random_kor_background(self, rng: random.Random, allowed_labels: list[str] | None = None) -> SampleRef:
        labels = allowed_labels or self.labels
        source = self.classes[rng.choice(labels)]
        refs = source.all_records()
        if not refs:
            raise ValueError("KorSynthetic contains an empty class folder")
        return rng.choice(refs)


def _find_cub_root(root: Path) -> Path:
    for candidate in (root, root / "CUB_200_2011"):
        if (candidate / "images.txt").exists() and (candidate / "images").is_dir():
            return candidate
    raise FileNotFoundError(f"CUB metadata/images not found under {root}")


def _read_id_path_file(path: Path) -> dict[int, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        result[int(key)] = value.strip()
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_kor_alphabet(root: str | Path) -> tuple[list[str], str]:
    """Read KorSynthetic's indexed alphabet and return its SHA-256 fingerprint."""
    root = Path(root).expanduser().resolve()
    parquet_manifest_path = root / "dataset_manifest.json"
    parquet_manifest = json.loads(parquet_manifest_path.read_text(encoding="utf-8")) if parquet_manifest_path.is_file() else None
    candidates = []
    if parquet_manifest and parquet_manifest.get("alphabet_file"):
        candidates.append(root / parquet_manifest["alphabet_file"])
    for parent in (root, root.parent):
        candidates.extend((parent / "kor_101_mfcr.json", parent / "alphabet.json"))
    alphabet_path = next((path for path in candidates if path.is_file()), None)
    if alphabet_path is None:
        raise FileNotFoundError(f"Cannot find kor_101_mfcr.json or alphabet.json under {root}")
    payload = json.loads(alphabet_path.read_text(encoding="utf-8"))
    alphabet = _alphabet_symbols(payload)
    return alphabet, _file_sha256(alphabet_path)


def _alphabet_symbols(payload: dict) -> list[str]:
    entries = payload.get("alphabet")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], list):
        raise ValueError("KorSynthetic alphabet file must contain an alphabet array")
    result = []
    for entry in entries[0]:
        if not isinstance(entry, list) or not entry or not isinstance(entry[0], str):
            raise ValueError("Unexpected KorSynthetic alphabet entry")
        result.append(entry[0])
    return result


def _numeric_class_root(root: Path) -> Path:
    for candidate in (root / "retrieve_mtr_per_class", root / "data", root):
        if candidate.is_dir() and any(path.is_dir() and path.name.isdigit() for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"Cannot find numeric KorSynthetic class folders under {root}")


def _read_parquet_dataset_manifest(root: Path, dataset: str) -> dict | None:
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "siamese-nets-parquet-v1":
        return None
    if payload.get("dataset") != dataset:
        raise ValueError(f"Parquet manifest at {root} is for {payload.get('dataset')!r}, not {dataset!r}")
    return payload


def _parquet_class_files(root: Path, manifest: dict) -> dict[str, list[Path]]:
    class_files = manifest.get("class_files")
    if not isinstance(class_files, dict):
        raise ValueError("Parquet dataset manifest has no class_files mapping")
    resolved = {}
    for class_index, files in class_files.items():
        paths = [root / relative for relative in files]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Parquet dataset is incomplete; missing shard {missing[0]}")
        resolved[str(class_index)] = paths
    return resolved


def load_catalog(
    name: str,
    root: str | Path,
    *,
    hangul_only: bool = True,
    phd_alphabet: list[str] | None = None,
    phd_alphabet_sha256: str | None = None,
) -> DatasetCatalog:
    root = Path(root).expanduser().resolve()
    name = name.lower()
    if name == "clinc150":
        data_path = root if root.is_file() else next(
            (path for path in (root / "data_full.json", root / "clinc150.json") if path.exists()), None
        )
        if data_path is None:
            raise FileNotFoundError(f"CLINC JSON not found under {root}")
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        classes: dict[str, list[SampleRef]] = {}
        for split in ("train", "val", "test", "oos_train", "oos_val", "oos_test"):
            for index, row in enumerate(payload[split]):
                text, label = row
                label = str(label)
                classes.setdefault(label, []).append(SampleRef(label, f"{split}:{index}", str(text), split))
        return DatasetCatalog(
            name, "text", {label: ClassSource(label, "text", records) for label, records in classes.items()},
            fingerprints={data_path.name: _file_sha256(data_path)},
        )

    if name == "cub200":
        data_root = _find_cub_root(root)
        labels_by_id = _read_id_path_file(data_root / "classes.txt")
        paths = _read_id_path_file(data_root / "images.txt")
        class_ids = _read_id_path_file(data_root / "image_class_labels.txt")
        official_splits = _read_id_path_file(data_root / "train_test_split.txt")
        classes: dict[str, list[SampleRef]] = {label: [] for label in labels_by_id.values()}
        split_map = {}
        for image_id, relative_path in paths.items():
            label = labels_by_id[int(class_ids[image_id])]
            uid = str(relative_path)
            classes[label].append(SampleRef(label, uid, str(data_root / "images" / relative_path)))
            split_map[uid] = "train" if int(official_splits[image_id]) == 1 else "test"
        sources = {
            label: ClassSource(label, "image", records, image_split_file=split_map)
            for label, records in classes.items()
        }
        hashes = {metadata: _file_sha256(data_root / metadata) for metadata in ("classes.txt", "images.txt", "image_class_labels.txt", "train_test_split.txt")}
        catalog = DatasetCatalog(name, "image", sources, fingerprints=hashes)
        catalog.image_protocol = "cub200_closed"
        return catalog

    if name == "omniglot":
        data_root = root / "python" if (root / "python").is_dir() else root
        classes: dict[str, list[SampleRef]] = {}
        for folder, source_split in (("images_background", "background"), ("images_evaluation", "evaluation")):
            source_root = data_root / folder
            if not source_root.is_dir():
                raise FileNotFoundError(f"Missing Omniglot folder: {source_root}")
            for alphabet in sorted(path for path in source_root.iterdir() if path.is_dir()):
                for character in sorted(path for path in alphabet.iterdir() if path.is_dir()):
                    label = f"{alphabet.name}/{character.name}"
                    records = [
                        SampleRef(label, f"{source_split}:{label}/{image.name}", str(image), source_split)
                        for image in sorted(character.iterdir()) if image.is_file() and image.suffix.lower() in (".png", ".jpg", ".jpeg")
                    ]
                    classes[label] = records
        catalog = DatasetCatalog(name, "image", {label: ClassSource(label, "image", records) for label, records in classes.items()})
        catalog.image_protocol = "omniglot"
        return catalog

    if name == "phd08":
        parquet_manifest = _read_parquet_dataset_manifest(root, "phd08")
        if parquet_manifest is not None:
            if (phd_alphabet is not None and parquet_manifest.get("alphabet_sha256") and
                    parquet_manifest["alphabet_sha256"] != phd_alphabet_sha256):
                raise ValueError("PHD08 Parquet catalog was converted with a different KorSynthetic alphabet")
            files_by_class = _parquet_class_files(root, parquet_manifest)
            class_indices = sorted((str(value) for value in parquet_manifest.get("class_indices", [])), key=lambda value: int(value))
            if not class_indices:
                raise ValueError("PHD08 Parquet manifest contains no class indices")
            sources = {}
            for class_index in class_indices:
                if class_index not in files_by_class:
                    raise ValueError(f"PHD08 Parquet manifest has no files for class {class_index}")
                label = class_index
                if phd_alphabet is not None:
                    if not class_index.isdigit() or int(class_index) >= len(phd_alphabet):
                        raise ValueError(f"PHD08 class index {class_index!r} is outside the KorSynthetic alphabet")
                    label = phd_alphabet[int(class_index)]
                    if len(label) != 1 or not 0xAC00 <= ord(label) <= 0xD7A3:
                        raise ValueError(f"PHD08 folder {class_index} maps to a non-Hangul symbol {label!r}")
                if label in sources:
                    raise ValueError(f"PHD08 class indices map to the same symbol {label!r}")
                sources[label] = ClassSource(
                    label, "image", parquet_files=files_by_class[class_index],
                    uid_prefix=class_index, family_grouping=True,
                )
            class_hash = hashlib.sha256("\n".join(class_indices).encode("utf-8")).hexdigest()
            fingerprints = {"class_names_sha256": class_hash}
            if phd_alphabet is not None:
                fingerprints["p5_catalog_labels_sha256"] = hashlib.sha256("\n".join(sorted(sources)).encode("utf-8")).hexdigest()
                if phd_alphabet_sha256 is not None:
                    fingerprints["alphabet_sha256"] = phd_alphabet_sha256
            catalog = DatasetCatalog(name, "image", sources, fingerprints=fingerprints)
            catalog.image_protocol = "stable_hash"
            return catalog
        data_root = root / "PHD08" if (root / "PHD08").is_dir() else root
        class_dirs = sorted(
            (path for path in data_root.iterdir() if path.is_dir()),
            key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name),
        )
        sources = {}
        for path in class_dirs:
            label = path.name
            if phd_alphabet is not None:
                if not path.name.isdigit() or int(path.name) >= len(phd_alphabet):
                    raise ValueError(f"PHD08 class folder {path.name!r} is not an index in the KorSynthetic alphabet")
                label = phd_alphabet[int(path.name)]
                if len(label) != 1 or not 0xAC00 <= ord(label) <= 0xD7A3:
                    raise ValueError(f"PHD08 folder {path.name} maps to a non-Hangul alphabet symbol {label!r}")
            if label in sources:
                raise ValueError(f"PHD08 folder indices do not map one-to-one at symbol {label!r}")
            sources[label] = ClassSource(label, "image", folder=path, uid_prefix=path.name, family_grouping=True)
        if not sources:
            raise FileNotFoundError(f"No PHD08 class folders under {data_root}")
        class_hash = hashlib.sha256("\n".join(path.name for path in class_dirs).encode("utf-8")).hexdigest()
        fingerprints = {"class_names_sha256": class_hash}
        if phd_alphabet is not None:
            mapped_labels = sorted(sources)
            fingerprints["p5_catalog_labels_sha256"] = hashlib.sha256("\n".join(mapped_labels).encode("utf-8")).hexdigest()
            if phd_alphabet_sha256 is not None:
                fingerprints["alphabet_sha256"] = phd_alphabet_sha256
        catalog = DatasetCatalog(name, "image", sources, fingerprints=fingerprints)
        catalog.image_protocol = "stable_hash"
        return catalog

    if name == "kor_synthetic":
        parquet_manifest = _read_parquet_dataset_manifest(root, "kor_synthetic")
        alphabet, alphabet_sha256 = load_kor_alphabet(root)
        sources = {}
        if parquet_manifest is not None:
            files_by_class = _parquet_class_files(root, parquet_manifest)
            folder_ids = sorted((int(value) for value in parquet_manifest.get("class_indices", [])))
            if len(folder_ids) != len(set(folder_ids)) or any(index < 0 or index >= len(alphabet) for index in folder_ids):
                raise ValueError("KorSynthetic Parquet manifest has invalid class indices")
            if not parquet_manifest.get("partial_dataset", False) and folder_ids != list(range(len(alphabet))):
                raise ValueError("Complete KorSynthetic Parquet catalog must contain every alphabet index")
            for class_index in folder_ids:
                if str(class_index) not in files_by_class:
                    raise ValueError(f"KorSynthetic Parquet manifest has no files for class {class_index}")
                label = alphabet[class_index]
                if hangul_only and not (0xAC00 <= ord(label) <= 0xD7A3):
                    continue
                sources[label] = ClassSource(
                    label, "kor_synthetic", parquet_files=files_by_class[str(class_index)],
                    uid_prefix=label,
                )
        else:
            class_root = _numeric_class_root(root)
            class_dirs = sorted((item for item in class_root.iterdir() if item.is_dir()), key=lambda item: int(item.name))
            folder_ids = [int(path.name) for path in class_dirs]
            if len(folder_ids) != len(set(folder_ids)):
                raise ValueError("KorSynthetic folders must have unique numeric indices")
            if set(folder_ids) != set(range(len(alphabet))):
                raise ValueError("KorSynthetic folders must cover the alphabet positions exactly")
            for path in class_dirs:
                class_index = int(path.name)
                if class_index >= len(alphabet):
                    raise ValueError(f"No alphabet symbol for KorSynthetic folder {path.name}")
                label = alphabet[class_index]
                if hangul_only and not (0xAC00 <= ord(label) <= 0xD7A3):
                    continue
                sources[label] = ClassSource(label, "kor_synthetic", folder=path)
        folder_names = [str(value) for value in sorted(int(index) for index in (parquet_manifest.get("class_indices", []) if parquet_manifest else folder_ids))]
        folder_hash = hashlib.sha256("\n".join(folder_names).encode("utf-8")).hexdigest()
        catalog = DatasetCatalog(name, "kor_synthetic", sources, fingerprints={
            "alphabet_sha256": alphabet_sha256,
            "class_folder_names_sha256": folder_hash,
        })
        catalog.image_protocol = "stable_hash"
        return catalog

    raise ValueError(f"Unknown dataset {name!r}; expected clinc150, cub200, omniglot, phd08, or kor_synthetic")


class SampleDataset(Dataset):
    def __init__(self, catalog: DatasetCatalog, label_to_id: dict[str, int], image_size: int = 224, train: bool = False, image_channels: int = 3, image_mean: list[float] | None = None, image_std: list[float] | None = None, background_labels: list[str] | None = None, background_seed: int | None = None):
        self.catalog = catalog
        self.label_to_id = label_to_id
        self.train = train
        self.image_channels = int(image_channels)
        self.background_labels = background_labels
        self.background_seed = int(background_seed) if background_seed is not None else None
        if train and catalog.name == "cub200":
            image_ops = [transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip()]
        else:
            image_ops = [transforms.Resize((image_size, image_size))]
        image_ops.append(transforms.ToTensor())
        if image_mean is not None and image_std is not None:
            image_ops.append(transforms.Normalize(mean=image_mean, std=image_std))
        self.image_transform = transforms.Compose(image_ops)
        self._background_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._parquet_row_group_cache: OrderedDict[tuple[str, int, tuple[str, ...] | None], object] = OrderedDict()

    def __len__(self):
        return 1

    def __getitem__(self, ref: SampleRef):
        label_id = self.label_to_id[ref.label]
        if self.catalog.modality == "text":
            return {"text": ref.value, "label": label_id, "class_name": ref.label, "uid": ref.uid}
        image = self._load_image(ref)
        return {"image": image, "label": label_id, "class_name": ref.label, "uid": ref.uid}

    def _load_image(self, ref: SampleRef) -> torch.Tensor:
        if self.catalog.modality == "kor_synthetic":
            if ref.parquet_path is not None:
                mask_values = self._kor_parquet_arrays(ref)["mask"]
            else:
                with open(ref.value, "r", encoding="utf-8") as stream:
                    source = json.load(stream)
                mask_values = source[1]["data"]
            mask = torch.as_tensor(mask_values, dtype=torch.float32).reshape(1, 37, 37)
            background_rng = random
            if self.background_seed is not None:
                digest = hashlib.blake2b(
                    f"{self.background_seed}:{ref.uid}".encode("utf-8"), digest_size=8
                ).digest()
                background_rng = random.Random(int.from_bytes(digest, "little"))
            background_ref = self.catalog.random_kor_background(background_rng, self.background_labels)
            cache_key = background_ref.uid
            if cache_key not in self._background_cache:
                if background_ref.parquet_path is not None:
                    background_data = self._kor_parquet_arrays(background_ref)["background"]
                else:
                    with open(background_ref.value, "r", encoding="utf-8") as stream:
                        background_data = json.load(stream)[0]["data"]
                self._background_cache[cache_key] = torch.as_tensor(background_data, dtype=torch.float32).reshape(1, 37, 37)
                while len(self._background_cache) > 64:
                    self._background_cache.popitem(last=False)
            image = self._background_cache[cache_key] * mask
            return image.expand(3, -1, -1) if self.image_channels == 3 else image

        if ref.parquet_path is not None:
            image_data = self._parquet_row(ref, columns=["image"])["image"]
            encoded = image_data.get("bytes") if isinstance(image_data, dict) else image_data
            if encoded is None:
                raise ValueError(f"Parquet image row has no embedded bytes: {ref.uid}")
            with Image.open(io.BytesIO(encoded)) as image_file:
                image = image_file.convert("L" if self.image_channels == 1 else "RGB")
                return self.image_transform(image)

        with Image.open(ref.value) as image_file:
            image = image_file.convert("L" if self.image_channels == 1 else "RGB")
            return self.image_transform(image)

    def _parquet_row(self, ref: SampleRef, columns: list[str] | None = None) -> dict:
        if ref.parquet_path is None or ref.parquet_row_group is None or ref.parquet_row_offset is None:
            raise ValueError(f"Incomplete Parquet row reference for {ref.uid}")
        selected_columns = tuple(columns) if columns is not None else None
        key = (ref.parquet_path, ref.parquet_row_group, selected_columns)
        if key in self._parquet_row_group_cache:
            self._parquet_row_group_cache.move_to_end(key)
            table = self._parquet_row_group_cache[key]
        else:
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("Reading Parquet datasets requires pyarrow; install requirements-experiments.txt") from exc
            table = pq.ParquetFile(ref.parquet_path).read_row_group(
                ref.parquet_row_group, columns=columns
            )
            self._parquet_row_group_cache[key] = table
            while len(self._parquet_row_group_cache) > 16:
                self._parquet_row_group_cache.popitem(last=False)
        return {
            name: table.column(name)[ref.parquet_row_offset].as_py()
            for name in table.column_names
        }

    def _kor_parquet_arrays(self, ref: SampleRef) -> dict:
        row = self._parquet_row(ref, columns=["background", "mask"])
        if "background" in row and "mask" in row:
            return row
        raise ValueError(f"KorSynthetic Parquet row is missing background/mask arrays: {ref.uid}")


def collate_samples(batch: list[dict]) -> dict:
    result = {
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "class_name": [item["class_name"] for item in batch],
        "uid": [item["uid"] for item in batch],
    }
    if "text" in batch[0]:
        result["text"] = [item["text"] for item in batch]
    else:
        result["image"] = torch.stack([item["image"] for item in batch])
    return result

