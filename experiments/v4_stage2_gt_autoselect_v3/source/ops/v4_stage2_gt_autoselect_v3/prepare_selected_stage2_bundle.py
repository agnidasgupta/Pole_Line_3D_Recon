#!/usr/bin/env python3
"""Create an experiment-only copy of the accepted V4 Stage-2 bundle.

Only the scalar line_threshold may be changed. The learned pole/line models and
feature columns are copied unchanged. The output bundle is a runtime artifact,
not a Git artifact, and should not be included in result archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_bundle", required=True)
    p.add_argument("--selection_json", required=True)
    p.add_argument("--output_bundle", required=True)
    p.add_argument("--output_metadata", required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    a = cli()
    inp = Path(a.input_bundle).resolve()
    sel = Path(a.selection_json).resolve()
    out = Path(a.output_bundle).resolve()
    meta = Path(a.output_metadata).resolve()
    if not inp.is_file():
        raise FileNotFoundError(inp)
    if not sel.is_file():
        raise FileNotFoundError(sel)
    decision = json.loads(sel.read_text())
    if not decision.get("selected"):
        raise RuntimeError(
            f"Selection did not approve a profile: {decision.get('selection_status')}"
        )
    profile = decision["selected_profile"]
    threshold = float(profile["line_refiner_threshold"])
    bundle = joblib.load(inp)
    required = {"pole_model", "line_model", "pole_threshold", "line_threshold"}
    missing = sorted(required - set(bundle))
    if missing:
        raise RuntimeError(f"Input bundle missing keys: {missing}")
    old_threshold = float(bundle["line_threshold"])
    derived = dict(bundle)
    derived["line_threshold"] = threshold
    derived["version"] = str(bundle.get("version", "v4_stage2")) + "+gt_autoselect_v3"
    derived["selection_metadata"] = {
        "selection_status": decision["selection_status"],
        "target_session": decision["target_session"],
        "selected_profile": profile,
        "source_bundle_sha256": sha256(inp),
        "source_line_threshold": old_threshold,
        "derived_line_threshold": threshold,
        "ground_truth_used_only_for_offline_selection": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    joblib.dump(derived, tmp)
    os.replace(tmp, out)
    check = joblib.load(out)
    if abs(float(check["line_threshold"]) - threshold) > 1e-12:
        raise RuntimeError("Derived bundle threshold verification failed")
    metadata = {
        "input_bundle": str(inp),
        "input_sha256": sha256(inp),
        "output_bundle": str(out),
        "output_sha256": sha256(out),
        "source_line_threshold": old_threshold,
        "selected_line_threshold": threshold,
        "models_copied_unchanged": True,
        "selection_json": str(sel),
    }
    atomic_json(metadata, meta)
    print("SELECTED_STAGE2_BUNDLE_OK")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
