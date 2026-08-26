#!/usr/bin/env python3
"""Export durable V4 Stage-1 artifacts to compact CSV.GZ and compute voxel metrics.

This is an operations/reporting utility. It does not run the model and does not
change production inference behavior.
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from v4_realtime_core import label_from_scores, load_calibration, row_level_scores
from v4_stage_contracts import atomic_json, load_stage1_artifact, safe_id

CLASSES = (0, 1, 2)
NAMES = {0: "background", 1: "pole", 2: "line"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--session_filter", required=True)
    p.add_argument("--calibration_json", required=True)
    p.add_argument("--resume", type=int, default=1)
    return p.parse_args()


def atom_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        df.to_csv(tmp, index=False, compression={"method": "gzip", "compresslevel": 1})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def has_label_column(source: str) -> bool:
    try:
        return "label" in pd.read_csv(source, nrows=0).columns
    except Exception:
        return False


def binary_metrics(y, p, positive):
    yb = y == positive
    pb = p == positive
    tp = int(np.sum(yb & pb)); fp = int(np.sum(~yb & pb)); fn = int(np.sum(yb & ~pb)); tn = int(np.sum(~yb & ~pb))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "iou": iou}


def multiclass_metrics(cm):
    rows = []
    for c in CLASSES:
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        rows.append({"class": c, "class_name": NAMES[c], "support": int(cm[c, :].sum()), "precision": precision, "recall": recall, "f1": f1, "iou": iou})
    return rows


def main():
    a = args(); root = Path(a.stage1_dir).resolve(); out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    mp = root / "stage1_manifest.csv"
    if not mp.is_file(): raise FileNotFoundError(mp)
    try: man = pd.read_csv(mp)
    except EmptyDataError as e: raise RuntimeError(f"Stage1 manifest is empty: {mp}") from e
    required = {"group_id", "slice_seq", "status", "relative_path", "stage1_npz", "stage1_meta_json", "source"}
    missing = sorted(required - set(man.columns))
    if missing: raise RuntimeError(f"Stage1 manifest missing columns {missing}: {mp}")
    man = man[(man.group_id.astype(str) == a.session_filter) & (man.status.astype(str) == "completed")].copy().sort_values("slice_seq")
    if man.empty: raise RuntimeError(f"No completed Stage1 rows for {a.session_filter}")
    cal = load_calibration(a.calibration_json)
    overall_cm = np.zeros((3, 3), dtype=np.int64)
    metric_rows = []; export_rows = []
    all_y = []; all_p = []
    for i, r in enumerate(man.itertuples(index=False), 1):
        npz = Path(str(r.stage1_npz)); meta = Path(str(r.stage1_meta_json))
        if not npz.is_file() or not meta.is_file():
            raise FileNotFoundError(f"Stage1 artifact missing for seq={r.slice_seq}: {npz} / {meta}")
        item, pred, md = load_stage1_artifact(npz, meta)
        rel = Path(str(r.relative_path)); stem = rel.name[:-7] if rel.name.endswith(".csv.gz") else rel.stem
        dest = out / "inference_rows" / rel.parent / f"{stem}_v4_inference.csv.gz"
        frame = row_level_scores(item, pred, int(getattr(r, "rows", len(item["coords"]))), cal)
        source_has_gt = has_label_column(str(r.source))
        if not source_has_gt and "label" in frame.columns:
            frame = frame.drop(columns=["label"])
        if not (a.resume and dest.is_file() and dest.stat().st_size):
            atom_csv_gz(frame, dest)
        export_rows.append({"group_id": a.session_filter, "slice_seq": int(r.slice_seq), "relative_path": str(r.relative_path), "rows_exported": len(frame), "inference_csv_gz": str(dest), "has_ground_truth": bool(source_has_gt)})
        mrow = {"group_id": a.session_filter, "slice_seq": int(r.slice_seq), "evaluated_rows": 0, "accuracy": math.nan, "has_ground_truth": bool(source_has_gt)}
        if source_has_gt:
            y = np.asarray(item["raw_labels"], dtype=np.int16)
            p = label_from_scores(pred["pole"], pred["line"], cal["pole_threshold"], cal["line_threshold"]).astype(np.int16)
            valid = np.isin(y, CLASSES)
            y = y[valid]; p = p[valid]
            cm = np.zeros((3, 3), dtype=np.int64)
            np.add.at(cm, (y, p), 1)
            overall_cm += cm; all_y.append(y); all_p.append(p)
            mrow.update({"evaluated_rows": int(len(y)), "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else math.nan})
            for c in (1, 2):
                bm = binary_metrics(y, p, c)
                for k, v in bm.items(): mrow[f"{NAMES[c]}_{k}"] = v
        metric_rows.append(mrow)
        print(f"[stage1-export] {i}/{len(man)} seq={int(r.slice_seq)} rows={len(frame)} gt={int(source_has_gt)} file={dest}", flush=True)
    pd.DataFrame(export_rows).to_csv(out / "inference_manifest.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "stage1_metrics_by_slice.csv", index=False)
    total = int(overall_cm.sum())
    payload = {
        "session": a.session_filter,
        "metric_scope": "occupied valid voxels only",
        "evaluated_rows": total,
        "accuracy": float(np.trace(overall_cm) / total) if total else None,
        "confusion_matrix_rows_true_cols_pred": overall_cm.tolist(),
        "classes": multiclass_metrics(overall_cm) if total else [],
    }
    if total:
        y = np.concatenate(all_y); p = np.concatenate(all_p)
        payload["pole_binary"] = binary_metrics(y, p, 1)
        payload["line_binary"] = binary_metrics(y, p, 2)
    atomic_json(payload, out / "stage1_metrics_summary.json")
    print("V4_STAGE1_EXPORT_AND_METRICS_OK", flush=True)

if __name__ == "__main__": main()
