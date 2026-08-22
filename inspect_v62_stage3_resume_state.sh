#!/usr/bin/env bash
set -euo pipefail
RUN="${RUN:-/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v62_teacher_recall}"
S3="${STAGE3:-$RUN/stage3_reconstruction}"
NAME="${NAME:-poleline-v62-teacher-reconstruction}"

echo "Stage3: $S3"
if sudo docker inspect "$NAME" >/dev/null 2>&1; then
  sudo docker inspect "$NAME" --format 'container={{.Name}} status={{.State.Status}} running={{.State.Running}} exit={{.State.ExitCode}} started={{.State.StartedAt}} finished={{.State.FinishedAt}}'
else
  echo "container=$NAME not present"
fi

complete=0
partial=0
if [[ -d "$S3/sessions" ]]; then
  while IFS= read -r -d '' d; do
    rel="${d#$S3/sessions/}"
    if [[ -s "$d/summary.json" && -s "$d/world_poles.csv" && -s "$d/conductor_chains.csv" && -s "$d/conductor_vertices.csv" && -s "$d/spans.csv" ]]; then
      complete=$((complete+1))
    elif find "$d" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .; then
      partial=$((partial+1))
      echo "PARTIAL  $rel"
    fi
  done < <(find "$S3/sessions" -mindepth 2 -maxdepth 2 -type d -print0 2>/dev/null | sort -z)
fi

echo "completed_sessions=$complete"
echo "partial_sessions=$partial"
echo "Recent completed summaries:"
find "$S3/sessions" -mindepth 3 -maxdepth 3 -type f -name summary.json -print 2>/dev/null | sort | tail -n 10 || true

echo "Recent Stage3 log tail:"
log=$(find "$S3" -maxdepth 1 -type f -name 'v62_stage3_clean_*.log' -print 2>/dev/null | sort | tail -n 1 || true)
if [[ -n "$log" ]]; then
  echo "$log"
  tail -n 30 "$log"
else
  echo "no stage3 log found"
fi
