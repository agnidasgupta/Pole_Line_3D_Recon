#!/usr/bin/env bash
set -euo pipefail
: "${SESSION_FILTER:?Set SESSION_FILTER=geography/sessionN}"
RUN=${RUN:-/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_realtime}
SAFE=$(printf '%s' "$SESSION_FILTER" | tr '/ ' '__')
REPLAY=${REPLAY:-$RUN/replays/$SAFE}
SHARE=${SHARE:-/tmp/v4_realtime_${SAFE}_diagnostics}
ARCHIVE=${ARCHIVE:-/tmp/v4_realtime_${SAFE}_diagnostics.tar.gz}
CODE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$SHARE" "$ARCHIVE"; mkdir -p "$SHARE"
python "$CODE/collect_v4_realtime_diagnostics.py" --run_root "$RUN" --replay_dir "$REPLAY" --output "$RUN/diagnostics/${SAFE}_realtime_diagnostics.json"
python3 - "$RUN" "$REPLAY" "$SHARE" "$SAFE" <<'PY'
from pathlib import Path
import json,shutil,sys
run=Path(sys.argv[1]); replay=Path(sys.argv[2]); dst=Path(sys.argv[3]); safe=sys.argv[4]
wanted=[
 'STAGE2_TRAINING_COMPLETED.json','diagnostics/runtime_variant_equivalence.json','diagnostics/v4_runtime_mode.env',f'diagnostics/{safe}_realtime_diagnostics.json',
 'stage2_mining/stage2_mining_runtime.csv','stage2_mining/target_counts.csv','stage2_mining/COMPLETED.json',
 'stage2_refiner/local_refiner_metrics.json','stage2_refiner/local_refiner_metrics.txt','stage2_refiner/pole_threshold_search.csv','stage2_refiner/line_threshold_search.csv','stage2_refiner/pole_feature_importance.csv','stage2_refiner/line_feature_importance.csv']
for rel in wanted:
 s=run/rel
 if s.is_file():
  d=dst/'run'/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,d)
for name in ['COMPLETED.json','REALTIME_REPLAY_VERIFICATION.json','realtime_slice_timing.csv','realtime_timing_summary.json','realtime_timing_summary.txt','inference_manifest.csv']:
 s=replay/name
 if s.is_file():
  d=dst/'replay'/name; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,d)
# Latest rolling Stage3 snapshot only. Full historical snapshots remain on Nebius/local full archive if needed.
latest_files=list((replay/'stage3_incremental').rglob('LATEST.json')) if (replay/'stage3_incremental').exists() else []
for lf in latest_files:
 try:
  meta=json.load(open(lf)); p=Path(meta['output_dir'])
  if str(p).startswith('/outputs/'):
   p=Path('/workspace/voxel_poleline/outputs')/str(p)[len('/outputs/'):]
  if p.is_dir(): shutil.copytree(p,dst/'latest_stage3_snapshot',dirs_exist_ok=True)
  shutil.copy2(lf,dst/'LATEST.json')
 except Exception as e: (dst/'LATEST_COPY_WARNING.txt').write_text(repr(e))
PY
tar -C "$(dirname "$SHARE")" -czf "$ARCHIVE" "$(basename "$SHARE")"
echo "Created $ARCHIVE"
ls -lh "$ARCHIVE"
