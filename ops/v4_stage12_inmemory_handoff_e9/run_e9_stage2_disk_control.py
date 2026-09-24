#!/usr/bin/env python3
"""Run the unchanged V10 disk-backed Stage-2 control with deterministic refiners.

Only worker count and timing observation are added.  Reconstruction code,
profile, calibration, Stage-1 artifact loading, and durable outputs are the
accepted V10 implementation.
"""
from __future__ import annotations

import time

import run_v4_stage2_stage1_electrical_tracks as stage2
from run_e9_stage12_inmemory import pin_refiner_workers


def main() -> None:
    original_factory = stage2.Stage1ElectricalTrackStage2Processor
    original_timing = stage2.upsert_timing
    last = time.perf_counter()

    def factory(*args, **kwargs):
        processor = original_factory(*args, **kwargs)
        pin_refiner_workers(processor)
        return processor

    def timing(path, row):
        nonlocal last
        now = time.perf_counter()
        observed = dict(row)
        observed["stage2_disk_slice_wall_ms"] = (now - last) * 1000.0
        original_timing(path, observed)
        last = time.perf_counter()

    stage2.Stage1ElectricalTrackStage2Processor = factory
    stage2.upsert_timing = timing
    try:
        stage2.main()
    finally:
        stage2.Stage1ElectricalTrackStage2Processor = original_factory
        stage2.upsert_timing = original_timing


if __name__ == "__main__":
    main()
