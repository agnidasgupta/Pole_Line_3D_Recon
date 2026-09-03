# V4 Stage2 Bundle-Consensus V7

Purpose: preserve accepted production Stage2 output exactly, while appending only novel Stage1-line-labelled residual conductor lanes that are supported by a multi-line production bundle.

Runtime invariants:
- Stage1 is not rerun.
- Production Stage2 components are preserved unchanged.
- GT is never read at runtime.
- No pole pair is enumerated.
- No synthetic line voxel is created.
- A residual lane must be backed by occupied Stage1 line-labelled voxels.
- A residual lane must have >=2 (or profile-specific >=3) parallel production siblings, longitudinal endpoint agreement, and bundle-consistent cross-section spacing.

Offline selector remains strict: precision-window false voxel/component/span deltas must not increase over production baseline.

Canonical output layout remains:
`.../v4_stage23_quality/full_run_v2_<UTC>/{stage2,stage3,selection,logs,timing,status}`

Files:
- v4_stage2_bundle_consensus.py
- select_v4_stage2_bundle_consensus_profile.py
- run_v4_stage2_bundle_consensus.py
- self_test_bundle_consensus.py
- run_v4_stage2_bundle_consensus_v7_on_nebius.sh
- monitor_v4_stage2_bundle_consensus_v7_on_nebius.sh
- download_v4_stage2_bundle_consensus_v7_to_mac.sh
