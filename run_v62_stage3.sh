#!/usr/bin/env bash
set -euo pipefail

INFERENCE_DIR="${INFERENCE_DIR:?set INFERENCE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
METADATA_DIR="${METADATA_DIR:-}"
CENTERS_CSV="${CENTERS_CSV:-}"
args=()
[[ -n "$METADATA_DIR" ]] && args+=(--metadata_dir "$METADATA_DIR")
[[ -n "$CENTERS_CSV" ]] && args+=(--centers_csv "$CENTERS_CSV")
[[ -n "${SESSION_FILTER:-}" ]] && args+=(--session_filter "$SESSION_FILTER")
[[ -n "${LATEST_SLICE:-}" ]] && args+=(--latest_slice "$LATEST_SLICE")
[[ -n "${FIXED_POLE_HEIGHT_FT:-}" ]] && args+=(--fixed_pole_height_ft "$FIXED_POLE_HEIGHT_FT")
[[ "${RESUME_STAGE3:-0}" == "1" ]] && args+=(--resume_sessions)

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/v62_stage3_clean_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

python reconstruct_v62_stage3.py \
  --inference_dir "$INFERENCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${args[@]}" \
  --world_units_to_ft "${WORLD_UNITS_TO_FT:-0.5}" \
  --pole_merge_radius_ft "${POLE_MERGE_RADIUS_FT:-4}" \
  --min_pole_separation_ft "${MIN_POLE_SEPARATION_FT:-10}" \
  --max_span_length_ft "${MAX_SPAN_LENGTH_FT:-450}" \
  --max_span_slices "${MAX_SPAN_SLICES:-9}" \
  --slice_length_ft "${SLICE_LENGTH_FT:-50}" \
  --fragment_join_radius_ft "${FRAGMENT_JOIN_RADIUS_FT:-14}" \
  --missing_slice_extra_ft "${MISSING_SLICE_EXTRA_FT:-8}" \
  --max_join_angle_deg "${MAX_JOIN_ANGLE_DEG:-18}" \
  --max_connector_angle_deg "${MAX_CONNECTOR_ANGLE_DEG:-25}" \
  --max_join_lateral_ft "${MAX_JOIN_LATERAL_FT:-1.5}" \
  --max_join_vertical_ft "${MAX_JOIN_VERTICAL_FT:-14}" \
  --max_join_vertical_horizontal_ratio "${MAX_JOIN_VERTICAL_HORIZONTAL_RATIO:-0.35}" \
  --max_z_extrap_error_ft "${MAX_Z_EXTRAP_ERROR_FT:-2.5}" \
  --max_longitudinal_overlap_ft "${MAX_LONGITUDINAL_OVERLAP_FT:-3.0}" \
  --overlap_xy_radius_ft "${OVERLAP_XY_RADIUS_FT:-2.0}" \
  --max_overlap_vertical_ft "${MAX_OVERLAP_VERTICAL_FT:-2.0}" \
  --fragment_bridge_radius_ft "${FRAGMENT_BRIDGE_RADIUS_FT:-30}" \
  --bridge_max_join_angle_deg "${BRIDGE_MAX_JOIN_ANGLE_DEG:-10}" \
  --bridge_max_connector_angle_deg "${BRIDGE_MAX_CONNECTOR_ANGLE_DEG:-15}" \
  --bridge_max_lateral_ft "${BRIDGE_MAX_LATERAL_FT:-1.0}" \
  --bridge_max_z_extrap_error_ft "${BRIDGE_MAX_Z_EXTRAP_ERROR_FT:-2.0}" \
  --self_intersection_tolerance_ft "${SELF_INTERSECTION_TOLERANCE_FT:-0.25}" \
  --pole_attachment_radius_ft "${POLE_ATTACHMENT_RADIUS_FT:-28}" \
  --pole_attachment_close_ft "${POLE_ATTACHMENT_CLOSE_FT:-4}" \
  --max_pole_attachment_angle_deg "${MAX_POLE_ATTACHMENT_ANGLE_DEG:-40}" \
  --max_pole_attachment_height_delta_ft "${MAX_POLE_ATTACHMENT_HEIGHT_DELTA_FT:-18}" \
  --pole_top_margin_ft "${POLE_TOP_MARGIN_FT:-1.5}" \
  --allowed_pole_height_variation_ft "${ALLOWED_POLE_HEIGHT_VARIATION_FT:-8}" \
  --max_pole_height_adjust_ft "${MAX_POLE_HEIGHT_ADJUST_FT:-8}" \
  --min_pole_height_ft "${MIN_POLE_HEIGHT_FT:-15}" \
  --pole_height_quantile_low "${POLE_HEIGHT_QUANTILE_LOW:-0.10}" \
  --pole_height_quantile_high "${POLE_HEIGHT_QUANTILE_HIGH:-0.90}" \
  --pole_height_range_margin_ft "${POLE_HEIGHT_RANGE_MARGIN_FT:-2.0}" \
  --pole_attachment_height_slack_ft "${POLE_ATTACHMENT_HEIGHT_SLACK_FT:-1.5}" \
  --span_completion_max_angle_deg "${SPAN_COMPLETION_MAX_ANGLE_DEG:-12}" \
  --span_completion_max_connector_angle_deg "${SPAN_COMPLETION_MAX_CONNECTOR_ANGLE_DEG:-15}" \
  --span_completion_max_lateral_ft "${SPAN_COMPLETION_MAX_LATERAL_FT:-1.5}" \
  --span_completion_max_z_error_ft "${SPAN_COMPLETION_MAX_Z_ERROR_FT:-3.0}" \
  --span_completion_max_gap_ft "${SPAN_COMPLETION_MAX_GAP_FT:-100}" \
  --span_completion_corridor_ft "${SPAN_COMPLETION_CORRIDOR_FT:-10}" \
  --span_completion_pole_extrap_ft "${SPAN_COMPLETION_POLE_EXTRAP_FT:-35}" \
  --span_completion_min_tracks "${SPAN_COMPLETION_MIN_TRACKS:-2}" \
  --span_completion_min_coverage "${SPAN_COMPLETION_MIN_COVERAGE:-0.15}" \
  --hidden_pole_min_line_angle_deg "${HIDDEN_POLE_MIN_LINE_ANGLE_DEG:-20}" \
  --hidden_pole_max_endpoint_extrap_ft "${HIDDEN_POLE_MAX_ENDPOINT_EXTRAP_FT:-20}" \
  --hidden_pole_ray_backtrack_tolerance_ft "${HIDDEN_POLE_RAY_BACKTRACK_TOLERANCE_FT:-1}" \
  --hidden_pole_max_attachment_z_spread_ft "${HIDDEN_POLE_MAX_ATTACHMENT_Z_SPREAD_FT:-15}" \
  --hidden_pole_slice_support_radius_ft "${HIDDEN_POLE_SLICE_SUPPORT_RADIUS_FT:-35}" \
  --hidden_pole_existing_pole_exclusion_ft "${HIDDEN_POLE_EXISTING_POLE_EXCLUSION_FT:-6}" \
  --hidden_pole_cluster_radius_ft "${HIDDEN_POLE_CLUSTER_RADIUS_FT:-4}" \
  --hidden_pole_ground_reference_radius_ft "${HIDDEN_POLE_GROUND_REFERENCE_RADIUS_FT:-150}" \
  --hidden_pole_min_track_evidence "${HIDDEN_POLE_MIN_TRACK_EVIDENCE:-2.0}" \
  --hidden_pole_require_distinct_anchor_poles "${HIDDEN_POLE_REQUIRE_DISTINCT_ANCHOR_POLES:-1}" \
  --smooth_spacing_ft "${SMOOTH_SPACING_FT:-2.0}"
