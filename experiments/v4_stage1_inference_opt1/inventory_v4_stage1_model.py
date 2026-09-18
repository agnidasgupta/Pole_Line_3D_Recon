#!/usr/bin/env python3
"""Emit the checkpoint-backed V4 Stage-1 architecture and parameter inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from v4_realtime_core import load_v4_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_txt", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model, cfg, compiled = load_v4_model(args.model_path, "cpu", False, "default")
    rows = []
    for name, module in model.named_modules():
        direct = list(module.named_parameters(recurse=False))
        if not direct:
            continue
        rows.append({
            "layer": name or "<root>",
            "type": type(module).__name__,
            "parameter_shapes": {key: list(value.shape) for key, value in direct},
            "trainable_parameters": int(sum(value.numel() for _, value in direct)),
        })

    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    payload = {
        "model_path": str(Path(args.model_path)),
        "compiled": bool(compiled),
        "checkpoint_config": cfg,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "layers": rows,
    }

    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "V4 STAGE1 CHECKPOINT MODEL INVENTORY",
        f"model_path={args.model_path}",
        f"checkpoint_config={json.dumps(cfg, sort_keys=True)}",
        f"total_parameters={total}",
        f"trainable_parameters={trainable}",
        "",
        "layer\ttype\tparameter_shapes\ttrainable_parameters",
    ]
    for row in rows:
        lines.append(
            f"{row['layer']}\t{row['type']}\t"
            f"{json.dumps(row['parameter_shapes'], sort_keys=True)}\t"
            f"{row['trainable_parameters']}"
        )
    output_txt.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "output_json": str(output_json),
        "output_txt": str(output_txt),
        "total_parameters": total,
        "trainable_parameters": trainable,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
