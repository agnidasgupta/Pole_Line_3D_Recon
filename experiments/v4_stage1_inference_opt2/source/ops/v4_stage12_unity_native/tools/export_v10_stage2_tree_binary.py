#!/usr/bin/env python3
"""Convert the exported ExtraTrees JSON into a compact Unity runtime asset.

This is an offline build-time tool. The resulting .bytes file is consumed by
native C# and requires no Python, joblib, sklearn, Docker, or network at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


MAGIC = b"V10ETB1\0"


def write_string(stream, value: str) -> None:
    data = value.encode("utf-8")
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)


def write_model(stream, model: dict) -> None:
    classes = [int(value) for value in model["classes"]]
    if 1 not in classes:
        raise RuntimeError(f"positive class 1 is absent: {classes}")
    positive = classes.index(1)
    trees = model["trees"]
    stream.write(struct.pack("<I", len(trees)))
    for tree in trees:
        left = [int(value) for value in tree["children_left"]]
        right = [int(value) for value in tree["children_right"]]
        feature = [int(value) for value in tree["feature"]]
        threshold = [float(value) for value in tree["threshold"]]
        values = tree["value"]
        count = len(left)
        if not (len(right) == len(feature) == len(threshold) == len(values) == count):
            raise RuntimeError("tree arrays are not aligned")
        probability = []
        for node in values:
            row = [float(value) for value in node]
            total = sum(row)
            value = row[positive] / total if total > 0.0 else 0.0
            if not math.isfinite(value):
                raise RuntimeError("non-finite tree probability")
            probability.append(value)
        stream.write(struct.pack("<I", count))
        stream.write(struct.pack(f"<{count}i", *left))
        stream.write(struct.pack(f"<{count}i", *right))
        stream.write(struct.pack(f"<{count}i", *feature))
        stream.write(struct.pack(f"<{count}d", *threshold))
        stream.write(struct.pack(f"<{count}d", *probability))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-bytes", required=True)
    args = parser.parse_args()

    source = Path(args.input_json).resolve()
    target = Path(args.output_bytes).resolve()
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).digest()
    data = json.loads(source_bytes)
    if int(data.get("format_version", -1)) != 1:
        raise RuntimeError("unsupported tree JSON format_version")
    features = [str(value) for value in data["feature_columns"]]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(MAGIC)
            stream.write(struct.pack("<I", 2))
            stream.write(source_sha256)
            stream.write(struct.pack("<I", len(features)))
            for feature in features:
                write_string(stream, feature)
            stream.write(struct.pack("<d", float(data["pole_threshold"])))
            stream.write(struct.pack("<d", float(data["line_threshold"])))
            write_model(stream, data["models"]["pole"])
            write_model(stream, data["models"]["line"])
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"V10_STAGE2_TREE_BINARY_OK path={target} bytes={target.stat().st_size}")


if __name__ == "__main__":
    main()
