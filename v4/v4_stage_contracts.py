#!/usr/bin/env python3
"""Durable contracts between V4 production stages.

The realtime orchestrator may pass objects in memory, but every stage also commits an
atomic on-disk representation.  This allows Stage 2 to be rerun from Stage 1 results,
and Stage 3 to be rerun from Stage 2 results, without re-executing earlier stages.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

CONTRACT_VERSION = "v4-production-1"

STAGE1_MANIFEST_COLUMNS = [
    "contract_version", "id", "source", "relative_path", "geography", "session",
    "slice_seq", "group_id", "center_x", "center_y", "center_z", "stage1_npz",
    "stage1_meta_json", "rows", "occupied_rows", "status",
]

STAGE2_MANIFEST_COLUMNS = [
    "contract_version", "id", "source", "relative_path", "geography", "session",
    "slice_seq", "group_id", "center_x", "center_y", "center_z", "stage1_npz",
    "output_csv", "pole_csv", "line_csv", "line_vertices_csv", "rows",
    "accepted_poles", "accepted_line_segments", "status",
]


def safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(s))


def atomic_json(obj, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_csv(df: pd.DataFrame, path, columns=None, **kwargs) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    if d.empty and columns is not None:
        d = pd.DataFrame(columns=columns)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    try:
        d.to_csv(tmp, index=False, **kwargs)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_npz(path, **arrays) -> None:
    """Write an uncompressed NPZ atomically; optimized for low-latency stage handoff."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def stage1_paths(root, relative_path: str):
    root = Path(root)
    rel = Path(relative_path)
    stem = rel.name[:-7] if rel.name.endswith(".csv.gz") else rel.stem
    base = root / "stage1_scores" / rel.parent
    return base / f"{stem}_stage1.npz", base / f"{stem}_stage1.json"


def stage2_paths(root, relative_path: str):
    root = Path(root)
    rel = Path(relative_path)
    stem = rel.name[:-7] if rel.name.endswith(".csv.gz") else rel.stem
    base = root / "stage2_objects" / rel.parent
    return (
        base / f"{stem}_poles.csv",
        base / f"{stem}_line_segments.csv",
        base / f"{stem}_line_vertices.csv",
    )


def save_stage1_artifact(npz_path, meta_path, item: Dict, pred: Dict, metadata: Dict) -> None:
    atomic_npz(
        npz_path,
        coords=np.asarray(item["coords"], dtype=np.int32),
        dist_values=np.asarray(item.get("dist_values", np.zeros(len(item["coords"]))), dtype=np.float32),
        source_rows=np.asarray(item.get("source_rows", np.arange(len(item["coords"]))), dtype=np.int64),
        raw_labels=np.asarray(item.get("raw_labels", np.zeros(len(item["coords"]))), dtype=np.int16),
        pole=np.asarray(pred["pole"], dtype=np.float32),
        line=np.asarray(pred["line"], dtype=np.float32),
        semantic=np.asarray(pred.get("semantic", np.zeros(len(item["coords"]))), dtype=np.uint8),
        objectness=np.asarray(pred.get("objectness", np.zeros(len(item["coords"]))), dtype=np.float32),
    )
    payload = dict(metadata)
    payload["contract_version"] = CONTRACT_VERSION
    payload["stage"] = 1
    payload["stage1_npz"] = str(Path(npz_path).resolve())
    atomic_json(payload, meta_path)


def load_stage1_artifact(npz_path, meta_path=None):
    p = Path(npz_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p) as d:
        required = {"coords", "pole", "line"}
        missing = sorted(required - set(d.files))
        if missing:
            raise RuntimeError(f"Stage-1 artifact missing arrays {missing}: {p}")
        coords = d["coords"].astype(np.int32, copy=True)
        n = len(coords)
        item = {
            "sparse_native": True,
            "coords": coords,
            "dist_values": d["dist_values"].astype(np.float32, copy=True) if "dist_values" in d else np.zeros(n, np.float32),
            "source_rows": d["source_rows"].astype(np.int64, copy=True) if "source_rows" in d else np.arange(n, dtype=np.int64),
            "raw_labels": d["raw_labels"].astype(np.int16, copy=True) if "raw_labels" in d else np.zeros(n, np.int16),
            "raw_hardneg": np.zeros(n, np.uint8),
            "z_sorted": coords[:, 2] if n else np.zeros(0, np.int32),
            "has_gt": "raw_labels" in d.files,
        }
        pred = {
            "pole": d["pole"].astype(np.float32, copy=True),
            "line": d["line"].astype(np.float32, copy=True),
            "semantic": d["semantic"].astype(np.uint8, copy=True) if "semantic" in d else np.zeros(n, np.uint8),
            "objectness": d["objectness"].astype(np.float32, copy=True) if "objectness" in d else np.zeros(n, np.float32),
            "timing": {},
        }
    if len(pred["pole"]) != n or len(pred["line"]) != n:
        raise RuntimeError(f"Stage-1 artifact coordinate/score length mismatch: {p}")
    meta = {}
    if meta_path and Path(meta_path).is_file():
        meta = json.loads(Path(meta_path).read_text())
        cv = meta.get("contract_version")
        if cv and cv != CONTRACT_VERSION:
            raise RuntimeError(f"Unsupported Stage-1 contract {cv}: {meta_path}")
    return item, pred, meta


def upsert_manifest_row(path, row: Dict, columns, key=("group_id", "slice_seq")) -> pd.DataFrame:
    p = Path(path)
    if p.is_file() and p.stat().st_size:
        try:
            d = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            d = pd.DataFrame(columns=columns)
        except Exception as exc:
            raise RuntimeError(f"Refusing to overwrite unreadable manifest {p}: {exc}") from exc
    else:
        d = pd.DataFrame(columns=columns)
    for c in columns:
        if c not in d.columns:
            d[c] = np.nan
    if not d.empty:
        mask = np.ones(len(d), dtype=bool)
        for k in key:
            if k == "slice_seq":
                vals = pd.to_numeric(d[k], errors="coerce")
                mask &= vals.eq(int(row[k])).to_numpy()
            else:
                mask &= d[k].astype(str).eq(str(row[k])).to_numpy()
        d = d.loc[~mask].copy()
    newrow = pd.DataFrame([row])
    d = newrow if d.empty else pd.concat([d, newrow], ignore_index=True)
    if "slice_seq" in d.columns:
        d["slice_seq"] = pd.to_numeric(d["slice_seq"], errors="coerce")
        d = d.sort_values(["group_id", "slice_seq"], kind="stable")
    atomic_csv(d.reindex(columns=columns), p, columns=columns)
    return d.reindex(columns=columns)
