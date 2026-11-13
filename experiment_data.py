"""Dataset catalogs shared by the experiment preparer, trainer, and evaluator.

Large directory datasets (PHD08 and KorSynthetic) are indexed by class folder
and read lazily. This avoids holding millions of Python path objects in RAM.
"""

from __future__ import annotations

import json
import hashlib
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
    ):
        if records is None and folder is None:
            raise ValueError("ClassSource needs records or a folder")
        self.label = label
        self.modality = modality
        self.records = records
        self.folder = folder
        self.image_split_file = image_split_file or {}
        self.uid_prefix = uid_prefix or label
        self._file_cache: OrderedDict[str, tuple[SampleRef, ...]] = OrderedDict()

    def _folder_records(self) -> tuple[SampleRef, ...]:
        key = str(self.folder)
        if key in self._file_cache:
            self._file_cache.move_to_end(key)
            return self._file_cache[key]
        assert self.folder is not None
        allowed = (".png", ".jpg", ".jpeg", ".bmp", ".json")
        with os.scandir(self.folder) as entries:
            names = sorted(
                entry.name for entry in entries
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(allowed)
            )
        records = tuple(SampleRef(self.label, f"{self.uid_prefix}/{name}", str(self.folder / name)) for name in names)
        self._file_cache[key] = records
        while len(self._file_cache) > 128:
            self._file_cache.popitem(last=False)
        return records

    def all_records(self) -> tuple[SampleRef, ...]:
        return tuple(self.records) if self.records is not None else self._folder_records()

    def records_for(
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
        eligible = self.records_for(split, seed, protocol)
        if max_examples is not None:
            # Stable ranking makes the 5-example set a subset of the 10-example
            # set when both use the same seed.
            eligible = tuple(sorted(eligible, key=lambda record: (stable_bucket(record.uid, seed + 7919), record.uid))[:max_examples])
        if len(eligible) < k:
            raise ValueError(f"Class {self.label!r} has {len(eligible)} eligible samples in {split}, needs {k}")
        return rng.sample(eligible, k)


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

    def random_kor_background(self, rng: random.Random, allowed_labels: list[str] | None = None) -> str:
        labels = allowed_labels or self.labels
        source = self.classes[rng.choice(labels)]
        refs = source.all_records()
        if not refs:
            raise ValueError("KorSynthetic contains an empty class folder")
        return rng.choice(refs).value


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
    class_root = root / "retrieve_mtr_per_class" if (root / "retrieve_mtr_per_class").is_dir() else root
    alphabet_path = root / "kor_101_mfcr.json" if (root / "kor_101_mfcr.json").exists() else class_root.parent / "kor_101_mfcr.json"
    payload = json.loads(alphabet_path.read_text(encoding="utf-8"))
    alphabet = [entry[0] for entry in payload["alphabet"][0]]
    return alphabet, _file_sha256(alphabet_path)


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
            sources[label] = ClassSource(label, "image", folder=path, uid_prefix=path.name)
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
        class_root = root / "retrieve_mtr_per_class" if (root / "retrieve_mtr_per_class").is_dir() else root
        alphabet, alphabet_sha256 = load_kor_alphabet(root)
        class_dirs = sorted((item for item in class_root.iterdir() if item.is_dir()), key=lambda item: int(item.name))
        folder_ids = [int(path.name) for path in class_dirs]
        if len(folder_ids) != len(set(folder_ids)):
            raise ValueError("KorSynthetic folders must have unique numeric indices")
        if set(folder_ids) != set(range(len(alphabet))):
            raise ValueError("KorSynthetic folder indices must cover the alphabet positions exactly")
        sources = {}
        for path in class_dirs:
            class_index = int(path.name)
            if class_index >= len(alphabet):
                raise ValueError(f"No alphabet symbol for KorSynthetic folder {path.name}")
            label = alphabet[class_index]
            if hangul_only and not (0xAC00 <= ord(label) <= 0xD7A3):
                continue
            sources[label] = ClassSource(label, "kor_synthetic", folder=path)
        folder_hash = hashlib.sha256("\n".join(path.name for path in class_dirs).encode("utf-8")).hexdigest()
        catalog = DatasetCatalog(name, "kor_synthetic", sources, fingerprints={
            "alphabet_sha256": alphabet_sha256,
            "class_folder_names_sha256": folder_hash,
        })
        catalog.image_protocol = "stable_hash"
        return catalog

    raise ValueError(f"Unknown dataset {name!r}; expected clinc150, cub200, omniglot, phd08, or kor_synthetic")


class SampleDataset(Dataset):
    def __init__(self, catalog: DatasetCatalog, label_to_id: dict[str, int], image_size: int = 224, train: bool = False, image_channels: int = 3, image_mean: list[float] | None = None, image_std: list[float] | None = None, background_labels: list[str] | None = None):
        self.catalog = catalog
        self.label_to_id = label_to_id
        self.train = train
        self.image_channels = int(image_channels)
        self.background_labels = background_labels
        if train and catalog.name == "cub200":
            image_ops = [transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip()]
        else:
            image_ops = [transforms.Resize((image_size, image_size))]
        image_ops.append(transforms.ToTensor())
        if image_mean is not None and image_std is not None:
            image_ops.append(transforms.Normalize(mean=image_mean, std=image_std))
        self.image_transform = transforms.Compose(image_ops)
        self._background_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

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
            with open(ref.value, "r", encoding="utf-8") as stream:
                source = json.load(stream)
            mask = torch.as_tensor(source[1]["data"], dtype=torch.float32).reshape(1, 37, 37)
            background_path = self.catalog.random_kor_background(random, self.background_labels)
            if background_path not in self._background_cache:
                with open(background_path, "r", encoding="utf-8") as stream:
                    background_data = json.load(stream)
                self._background_cache[background_path] = torch.as_tensor(background_data[0]["data"], dtype=torch.float32).reshape(1, 37, 37)
                while len(self._background_cache) > 64:
                    self._background_cache.popitem(last=False)
            image = self._background_cache[background_path] * mask
            return image.expand(3, -1, -1) if self.image_channels == 3 else image

        with Image.open(ref.value) as image_file:
            image = image_file.convert("L" if self.image_channels == 1 else "RGB")
            return self.image_transform(image)


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

