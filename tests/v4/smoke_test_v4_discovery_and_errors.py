#!/usr/bin/env python3
"""Synthetic production error-path tests. No GPU/model files required."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_v4_realtime_session import discover
from v4_realtime_core import build_sparse_item_from_dataframe
from v4_stage_contracts import (
    STAGE1_MANIFEST_COLUMNS, atomic_csv, load_stage1_artifact, save_stage1_artifact,
    stage1_paths, stage2_paths, upsert_manifest_row,
)


def expect(exc_type, fn, contains: str = ""):
    try:
        fn()
    except exc_type as exc:
        if contains and contains not in str(exc):
            raise AssertionError(f"Expected {contains!r} in {exc!r}") from exc
        return
    raise AssertionError(f"Expected {exc_type.__name__}")


def write_slice(root: Path, geo: str, session: int, seq: int, name: str | None = None):
    d = root / geo / f"session{session}_slice{seq}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / (name or f"slice_{seq}.csv")
    pd.DataFrame({
        "x":[0,1,1,np.nan,999], "y":[0,1,1,2,2], "z":[0,1,1,2,2],
        "label":[0,1,2,0,0], "dist_center_ft":[0,2,4,1,1],
        "center_x":[10]*5, "center_y":[20]*5, "center_z":[30]*5,
    }).to_csv(p,index=False)
    return p


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "voxel_csv_combined"; root.mkdir()
        write_slice(root,"GeoA",3,202)
        write_slice(root,"GeoA",3,204)  # gap is legal; missing slice 203 is not an error.
        rows = discover(root,"GeoA/session3")
        assert [x[0] for x in rows] == [202,204]
        expect(FileNotFoundError, lambda: discover(root,"Missing/session3"), "Missing")
        expect(ValueError, lambda: discover(root,"GeoA"), "geography/sessionN")
        expect(ValueError, lambda: discover(root,"GeoA/notasession"), "geography/sessionN")

        # A second CSV source for the same sequence must fail rather than silently
        # choosing one and corrupting realtime order.
        write_slice(root,"GeoA",3,202,"duplicate.csv")
        expect(RuntimeError, lambda: discover(root,"GeoA/session3"), "Multiple CSV sources")

    df = pd.DataFrame({
        "x":[1,1,2,np.nan,401], "y":[1,1,2,1,1], "z":[1,1,2,1,1],
        "label":[0,2,1,1,1], "dist_center_ft":[1,3,2,4,5],
    })
    item = build_sparse_item_from_dataframe(df,(400,400,200))
    # Duplicate coordinate keeps the last valid record; NaN/out-of-grid rows are dropped.
    assert len(item["coords"]) == 2
    assert item["coords"].shape == (2,3)
    expect(ValueError, lambda: build_sparse_item_from_dataframe(pd.DataFrame({"x":[1],"y":[1]})), "missing required local columns")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        npz, meta = stage1_paths(root,"GeoA/session3_slice202/slice_202.csv")
        pred = {"pole":np.array([.1,.2],np.float32), "line":np.array([.3,.4],np.float32),
                "semantic":np.array([0,2],np.uint8), "objectness":np.array([.5,.6],np.float32)}
        md = {"id":"x","rows":5,"occupied_rows":2,"center_metadata":{"center_x":10.,"center_y":20.,"center_z":30.}}
        save_stage1_artifact(npz,meta,item,pred,md)
        item2,pred2,md2 = load_stage1_artifact(npz,meta)
        assert np.array_equal(item2["coords"],item["coords"])
        assert np.allclose(pred2["pole"],pred["pole"])
        assert md2["center_metadata"]["center_x"] == 10.0
        p,l,v = stage2_paths(root,"GeoA/session3_slice202/slice_202.csv.gz")
        assert p.name == "slice_202_poles.csv" and l.name == "slice_202_line_segments.csv" and v.name == "slice_202_line_vertices.csv"

        # Corrupt/incompatible artifact must fail loudly.
        bad = root/"bad.npz"; np.savez(bad,coords=np.zeros((1,3),np.int32),pole=np.zeros(1,np.float32))
        expect(RuntimeError, lambda: load_stage1_artifact(bad), "missing arrays")

        manifest = root/"stage1_manifest.csv"
        row = {c:"" for c in STAGE1_MANIFEST_COLUMNS}; row.update({"group_id":"GeoA/session3","slice_seq":202,"status":"completed"})
        upsert_manifest_row(manifest,row,STAGE1_MANIFEST_COLUMNS)
        # A malformed nonempty manifest is never silently overwritten.
        manifest.write_text('"unterminated\n')
        expect(RuntimeError, lambda: upsert_manifest_row(manifest,row,STAGE1_MANIFEST_COLUMNS), "Refusing to overwrite unreadable manifest")

    print("V4_DISCOVERY_ERROR_PATHS_OK gaps=allowed duplicate_sequence=rejected malformed_manifest=rejected relocated_paths=covered")


if __name__ == "__main__":
    main()
