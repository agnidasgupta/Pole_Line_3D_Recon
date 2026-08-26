#!/usr/bin/env bash
# Create a compact ChatGPT-review archive from the large full-dataset results tarball.
# Mac-side only; uses tar/grep/awk/sed/sort/shasum and never invokes Python.
set -euo pipefail

ARCH=${1:-}
if [[ -z "$ARCH" ]]; then
  ARCH=$(find "$HOME/Downloads/v4_all_data_results" -type f -name 'v4_all_data_results_*.tar.gz' -print 2>/dev/null | sort | tail -1 || true)
fi
[[ -n "$ARCH" && -s "$ARCH" ]] || { echo "ERROR: full-dataset result archive not found. Pass it as argument 1." >&2; exit 2; }

OUT_ROOT=${2:-$HOME/Downloads/v4_compact_review}
REVIEW_MODE=${3:-full}
case "$REVIEW_MODE" in full|metrics) ;; *) echo "ERROR: review mode must be full or metrics" >&2; exit 2;; esac
mkdir -p "$OUT_ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BASE=$(basename "$ARCH" .tar.gz)
OUT="$OUT_ROOT/${BASE}_compact_review_${STAMP}.tar.gz"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
MEMBERS="$TMP/members.txt"
SELECT="$TMP/select.txt"
EXTRACT="$TMP/extract"
mkdir -p "$EXTRACT"

tar -tzf "$ARCH" > "$MEMBERS"
: > "$SELECT"

# Core summaries, context, metrics and timings.
grep -v '/$' "$MEMBERS" | grep -E '^\./(PACKAGE_INFO\.txt|COMPLETED\.txt|FULL_DATASET_SUMMARY\.(json|txt)|context/.*|metrics/.*|timings/.*)$' >> "$SELECT" || true

# Durable Stage 1/2 manifests are useful for counts and session/slice checks.
grep -v '/$' "$MEMBERS" | grep -E '/(stage1_manifest|stage2_manifest|inference_manifest)\.csv$' >> "$SELECT" || true

# Optional reconstruction context. "full" includes all Stage3 summary JSONs and
# the final Stage3 reconstruction directory for every session. "metrics" omits them.
if [[ "$REVIEW_MODE" == "full" ]]; then
  grep -v '/$' "$MEMBERS" | grep -E '^\./stage3/.*/summary\.json$' >> "$SELECT" || true

  FINAL="$TMP/final_summaries.tsv"
  grep -E '^\./stage3/.*/stage3_incremental/.*/slice_[0-9]+/sessions/.*/summary\.json$' "$MEMBERS" \
    | awk '
        {
          p=$0
          if (match(p,/\/slice_[0-9]+\//)) {
            pre=substr(p,1,RSTART-1)
            s=substr(p,RSTART+7,RLENGTH-8)+0
            print pre "\t" s "\t" p
          }
        }
      ' \
    | sort -t $'\t' -k1,1 -k2,2n \
    | awk -F '\t' '{last[$1]=$3} END{for(k in last) print last[k]}' > "$FINAL" || true

  while IFS= read -r summary; do
    [[ -n "$summary" ]] || continue
    dir=${summary%/summary.json}
    grep -v '/$' "$MEMBERS" | grep -F "$dir/" >> "$SELECT" || true
  done < "$FINAL"
fi

sort -u "$SELECT" -o "$SELECT"
[[ -s "$SELECT" ]] || { echo "ERROR: no review files selected from $ARCH" >&2; exit 3; }

if tar --help 2>&1 | grep -F -- "--verbatim-files-from" >/dev/null; then
  tar --verbatim-files-from -xzf "$ARCH" -C "$EXTRACT" -T "$SELECT"
else
  tar -xzf "$ARCH" -C "$EXTRACT" -T "$SELECT"
fi
cat > "$EXTRACT/COMPACT_REVIEW_INFO.txt" <<INFO
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_archive=$ARCH
selected_member_count=$(wc -l < "$SELECT" | tr -d ' ')
review_mode=$REVIEW_MODE
selection=summary_context_metrics_timings_manifests_plus_optional_stage3_context
excluded=per_voxel_stage1_inference_csv_gz_and_nonfinal_stage3_bulk
INFO

tar -C "$EXTRACT" -czf "$OUT" .
SHA=$(shasum -a 256 "$OUT" | awk '{print $1}')
printf '%s  %s\n' "$SHA" "$(basename "$OUT")" > "$OUT.sha256"
echo "Compact review archive: $OUT"
echo "SHA256: $SHA"
echo "Size: $(du -h "$OUT" | awk '{print $1}')"
