#!/usr/bin/env python3
"""Instrument-only wrapper around the accepted disk-backed V10 Stage-2 runner."""
from __future__ import annotations

import time
from typing import Any

import run_v4_stage2_stage1_electrical_tracks as stage2


def pin_refiner_workers(root: Any) -> int:
    """Pin only the loaded Stage-2 refiners; avoid importing Stage-1/E9 here."""
    visited: set[int] = set()
    changed = 0

    def walk(obj: Any, depth: int = 0) -> None:
        nonlocal changed
        if obj is None or depth > 6 or id(obj) in visited:
            return
        visited.add(id(obj))
        if hasattr(obj, "n_jobs"):
            try:
                if getattr(obj, "n_jobs") != 1:
                    setattr(obj, "n_jobs", 1)
                    changed += 1
            except Exception:
                pass
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value, depth + 1)
        elif hasattr(obj, "__dict__") and obj.__class__.__module__.split(".")[0] not in {"numpy", "pandas", "torch"}:
            for value in vars(obj).values():
                walk(value, depth + 1)

    walk(root)
    return changed


def main() -> None:
    original_load = stage2.load_stage1_artifact
    original_factory = stage2.Stage1ElectricalTrackStage2Processor
    original_csv = stage2.atomic_csv
    original_json = stage2.atomic_json
    original_manifest = stage2.upsert_manifest_row
    original_timing = stage2.upsert_timing
    current: dict[str, float] = {}
    slice_started = 0.0

    def add(name: str, start: float) -> None:
        current[name] = current.get(name, 0.0) + (time.perf_counter() - start) * 1000.0

    def load(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original_load(*args, **kwargs)
        finally:
            add("e11_disk_artifact_load_ms", start)

    class ProcessorProxy:
        def __init__(self, processor: Any) -> None:
            self._processor = processor

        def process(self, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return self._processor.process(*args, **kwargs)
            finally:
                add("e11_disk_process_ms", start)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._processor, name)

    def factory(*args: Any, **kwargs: Any) -> ProcessorProxy:
        return ProcessorProxy(pin_and_return(original_factory(*args, **kwargs)))

    def pin_and_return(processor: Any) -> Any:
        pin_refiner_workers(processor)
        return processor

    def csv(frame: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original_csv(frame, path, *args, **kwargs)
        finally:
            name = str(path)
            if name.endswith(("_poles.csv", "_lines.csv", "_line_vertices.csv", "_components.csv")):
                bucket = "e11_disk_primary_output_write_ms"
            elif name.endswith(("inference_manifest.csv", "stage2_manifest.csv")):
                bucket = "e11_disk_manifest_update_ms"
            else:
                bucket = "e11_disk_audit_output_write_ms"
            add(bucket, start)

    def js(obj: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original_json(obj, path, *args, **kwargs)
        finally:
            add("e11_disk_audit_json_write_ms", start)

    def manifest(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original_manifest(*args, **kwargs)
        finally:
            add("e11_disk_manifest_update_ms", start)

    def timing(path: Any, row: dict[str, Any]) -> None:
        nonlocal current, slice_started
        elapsed_before_write = (time.perf_counter() - slice_started) * 1000.0
        observed = dict(row)
        observed.update(current)
        observed["e11_disk_callback_ms_excluding_timing_write"] = elapsed_before_write
        observed["e11_disk_unattributed_ms"] = elapsed_before_write - sum(current.values())
        start = time.perf_counter()
        original_timing(path, observed)
        timing_write = (time.perf_counter() - start) * 1000.0
        # This field is reported on the following timing update in memory only;
        # it is deliberately not rewritten, preserving the accepted output row.
        current = {"e11_disk_timing_row_write_previous_ms": timing_write}
        slice_started = time.perf_counter()

    stage2.load_stage1_artifact = load
    stage2.Stage1ElectricalTrackStage2Processor = factory
    stage2.atomic_csv = csv
    stage2.atomic_json = js
    stage2.upsert_manifest_row = manifest
    stage2.upsert_timing = timing
    slice_started = time.perf_counter()
    try:
        stage2.main()
    finally:
        stage2.load_stage1_artifact = original_load
        stage2.Stage1ElectricalTrackStage2Processor = original_factory
        stage2.atomic_csv = original_csv
        stage2.atomic_json = original_json
        stage2.upsert_manifest_row = original_manifest
        stage2.upsert_timing = original_timing


if __name__ == "__main__":
    main()
