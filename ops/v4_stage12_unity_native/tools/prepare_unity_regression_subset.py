#!/usr/bin/env python3
"""Build a deterministic manifest/reference subset without numpy or pandas."""
import argparse
import csv
import re
import shutil
from pathlib import Path


def sid(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def stem(value: str) -> str:
    name = Path(value).name
    return name[:-7] if name.endswith(".csv.gz") else Path(name).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-root", required=True)
    parser.add_argument("--accepted-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--group-id", default="VELASCO_CUT_CP/session1")
    parser.add_argument("--ordinal-min", type=int, default=20)
    parser.add_argument("--ordinal-max", type=int, default=39)
    args = parser.parse_args()

    stage1 = Path(args.stage1_root).resolve()
    accepted = Path(args.accepted_root).resolve()
    output = Path(args.output_root).resolve()
    rows = []
    fields = []
    for manifest in sorted(stage1.rglob("stage1_manifest.csv")):
        with manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields.extend(reader.fieldnames or [])
            for row in reader:
                if row.get("group_id") == args.group_id and row.get("status", "completed") == "completed":
                    rows.append(row)
    rows.sort(key=lambda row: int(row["slice_seq"]))
    chosen = rows[args.ordinal_min:args.ordinal_max + 1]
    expected = args.ordinal_max - args.ordinal_min + 1
    if len(chosen) != expected:
        raise SystemExit(f"expected {expected} regression rows, found {len(chosen)}")

    manifest_out = output / "input_manifest.csv"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(fields + ["source_csv"]))
    with manifest_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in chosen:
            row = dict(row)
            row["source_csv"] = row.get("source_csv") or row.get("source", "")
            if not row["source_csv"]:
                raise SystemExit("selected row lacks source/source_csv")
            writer.writerow(row)

    session = sid(args.group_id)
    source_objects = accepted / "stage2" / session / "stage2_objects"
    target_objects = output / "reference" / "stage2" / session / "stage2_objects"
    copied = 0
    for row in chosen:
        relative = Path(row["relative_path"])
        source_dir = source_objects / relative.parent
        target_dir = target_objects / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in source_dir.glob(stem(row["relative_path"]) + "_*"):
            if path.is_file():
                shutil.copy2(path, target_dir / path.name)
                copied += 1
    if copied == 0:
        raise SystemExit("no accepted Stage2 reference files copied")
    print(f"V10_UNITY_REGRESSION_SUBSET_OK rows={len(chosen)} files={copied} root={output}")


if __name__ == "__main__":
    main()
