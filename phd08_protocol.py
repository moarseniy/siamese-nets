"""Filename-derived PHD08 source-family keys and deterministic group splits."""

from __future__ import annotations

import hashlib
from pathlib import Path


PHD08_FAMILY_VARIANTS_PER_BASE = 9
PHD08_BASE_FAMILIES_PER_CLASS = 243


def phd08_family_id(class_folder: str, filename: str) -> str:
    """Use the PHD08 class folder and base-render prefix as one family ID.

    PHD08 filenames have a font code and five ternary setting fields. The first
    four stem fields identify a base render; the last two fields enumerate its
    nine threshold/rotation variants.
    """
    fields = Path(filename).stem.split("_")
    if len(fields) != 6 or not fields[0] or any(value not in {"0", "1", "2"} for value in fields[1:]):
        raise ValueError(f"Unexpected PHD08 filename format for family grouping: {filename!r}")
    return f"{class_folder}/{'_'.join(fields[:4])}"


def _family_rank(family_id: str, seed: int) -> tuple[bytes, str]:
    digest = hashlib.blake2b(f"{int(seed)}:{family_id}".encode("utf-8"), digest_size=8).digest()
    return digest, family_id


def phd08_family_splits(family_ids: list[str] | tuple[str, ...], seed: int) -> dict[str, set[str]]:
    """Split 243 groups per class into exact 60/20/20 and 50/50 partitions."""
    families = sorted(set(family_ids), key=lambda family: _family_rank(family, seed))
    if len(families) != PHD08_BASE_FAMILIES_PER_CLASS:
        raise ValueError(
            f"Expected {PHD08_BASE_FAMILIES_PER_CLASS} PHD08 source families in a class; "
            f"found {len(families)}"
        )
    train_count = int(0.6 * len(families) + 0.5)
    valid_count = int(0.2 * len(families) + 0.5)
    gallery_count = (len(families) + 1) // 2
    return {
        "train": set(families[:train_count]),
        "valid": set(families[train_count:train_count + valid_count]),
        "test": set(families[train_count + valid_count:]),
        "gallery": set(families[:gallery_count]),
        "query": set(families[gallery_count:]),
    }
