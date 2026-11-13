"""Create or validate the explicit PHD08 -> KorSynthetic symbol mapping.

PHD08 directory numbers are not assumed to be Unicode indices. Fill the
template from a documented source before enabling the P5 cross-dataset run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phd-manifest", type=Path, default=Path("experiments/manifests/phd08.json"))
    parser.add_argument("--kor-manifest", type=Path, default=Path("experiments/manifests/kor_synthetic.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/manifests/phd08_kor_mapping.csv"))
    parser.add_argument("--validate", action="store_true", help="Validate the completed CSV instead of writing a blank template")
    args = parser.parse_args()
    phd = json.loads(args.phd_manifest.read_text(encoding="utf-8"))
    kor = json.loads(args.kor_manifest.read_text(encoding="utf-8"))
    phd_classes = phd["protocols"]["closed_set"]["classes"]
    kor_symbols = set(kor["protocols"]["hangul_only"]["classes"])
    folder_for_symbol = {symbol: folder for folder, symbol in kor["class_folder_to_symbol"].items()}

    if not args.validate:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("phd08_class_folder", "hangul_symbol"))
            writer.writerows((label, "") for label in phd_classes)
        print(f"Wrote mapping template with {len(phd_classes)} PHD08 classes: {args.output}")
        print("No class-code correspondence was inferred.")
        return

    rows = list(csv.DictReader(args.output.open(encoding="utf-8", newline="")))
    mapping = {row["phd08_class_folder"]: row["hangul_symbol"] for row in rows if row.get("hangul_symbol")}
    missing = sorted(set(phd_classes) - set(mapping))
    extra = sorted(set(mapping) - set(phd_classes))
    invalid = sorted((label, symbol) for label, symbol in mapping.items() if symbol not in kor_symbols)
    duplicate_symbols = sorted(symbol for symbol in set(mapping.values()) if list(mapping.values()).count(symbol) > 1)
    duplicate_source_rows = len(rows) != len({row["phd08_class_folder"] for row in rows})
    if missing or extra or invalid or duplicate_symbols or duplicate_source_rows:
        raise ValueError(
            f"Invalid PHD08 mapping: missing={len(missing)}, extra={len(extra)}, "
            f"not_in_KorSynthetic={len(invalid)}, duplicate_symbols={len(duplicate_symbols)}, "
            f"duplicate_source_rows={duplicate_source_rows}"
        )
    resolved = {label: {"symbol": symbol, "kor_synthetic_folder": folder_for_symbol[symbol]}
                for label, symbol in sorted(mapping.items())}
    output_json = args.output.with_suffix(".json")
    output_json.write_text(json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated one-to-one mapping for {len(resolved)} classes: {output_json}")


if __name__ == "__main__":
    main()

