#!/usr/bin/env bash
set -euo pipefail
: "${SESSION_FILTER:?Set SESSION_FILTER=geography/sessionN}"
RUN=${RUN:-/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_realtime}
SAFE=$(printf '%s' "$SESSION_FILTER" | tr '/ ' '__')
REPLAY=${REPLAY:-$RUN/replays/$SAFE}
if [[ ! -d "$REPLAY" ]]; then
  echo "ERROR: replay directory not found: $REPLAY" >&2
  echo "Set REPLAY=/workspace/.../the_actual_replay_or_benchmark_directory" >&2
  exit 2
fi
SHARE=${SHARE:-/tmp/v4_production_review_${SAFE}}
ARCHIVE=${ARCHIVE:-/tmp/v4_production_review_${SAFE}.tar.gz}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rm -rf "$SHARE" "$ARCHIVE"; mkdir -p "$SHARE"
# Generate current summaries without changing inference/reconstruction.
python3 "$CODE/collect_v4_realtime_diagnostics.py" --run_root "$RUN" --replay_dir "$REPLAY" --output "$REPLAY/realtime_diagnostics.json"
python3 "$CODE/summarize_v4_realtime_timing.py" --timing_csv "$REPLAY/realtime_slice_timing.csv" --output_json "$REPLAY/realtime_timing_summary.json" --output_txt "$REPLAY/realtime_timing_summary.txt"
python3 - "$RUN" "$REPLAY" "$SHARE" <<'PY'
from pathlib import Path
import json, shutil, sys
run=Path(sys.argv[1]); replay=Path(sys.argv[2]); dst=Path(sys.argv[3])
run_files=[
 'diagnostics/runtime_variant_equivalence.json','diagnostics/v4_runtime_mode.env',
 'STAGE2_TRAINING_COMPLETED.json','stage2_mining/COMPLETED.json','stage2_mining/stage2_mining_runtime.csv','stage2_mining/target_counts.csv',
 'stage2_refiner/local_refiner_metrics.json','stage2_refiner/local_refiner_metrics.txt',
 'stage2_refiner/pole_threshold_search.csv','stage2_refiner/line_threshold_search.csv',
 'stage2_refiner/pole_feature_importance.csv','stage2_refiner/line_feature_importance.csv']
replay_files=['COMPLETED.json','REALTIME_REPLAY_VERIFICATION.json','realtime_slice_timing.csv','realtime_timing_summary.json','realtime_timing_summary.txt','realtime_diagnostics.json','inference_manifest.csv']
for rel in run_files:
    s=run/rel
    if s.is_file():
        d=dst/'run'/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,d)
# Include Stage-1 batch-sweep summaries, whichever gated runtime mode was tested.
for sweep in (run/'diagnostics').glob('batch_sweep_*') if (run/'diagnostics').exists() else []:
    if sweep.is_dir():
        for name in ('batch_size_summary.csv','batch_size_summary.json'):
            s=sweep/name
            if s.is_file():
                d=dst/'run'/'diagnostics'/sweep.name/name; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,d)
for rel in replay_files:
    s=replay/rel
    if s.is_file():
        d=dst/'replay'/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,d)
# Copy the latest Stage-3 snapshot only for compact review.
latest=list((replay/'stage3_incremental').rglob('LATEST.json')) if (replay/'stage3_incremental').exists() else []
for lf in latest:
    try:
        meta=json.load(open(lf)); p=Path(meta['output_dir'])
        if str(p).startswith('/outputs/'):
            p=Path('/workspace/voxel_poleline/outputs')/str(p)[len('/outputs/'):]
        if p.is_dir(): shutil.copytree(p,dst/'latest_stage3_snapshot',dirs_exist_ok=True)
        shutil.copy2(lf,dst/'LATEST.json')
    except Exception as e:
        (dst/'LATEST_COPY_WARNING.txt').write_text(repr(e))
PY
tar -C "$(dirname "$SHARE")" -czf "$ARCHIVE" "$(basename "$SHARE")"
echo "Created $ARCHIVE"
ls -lh "$ARCHIVE"
