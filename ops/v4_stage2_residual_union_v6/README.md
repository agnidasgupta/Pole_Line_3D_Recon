# V4 Stage2 Residual-Union V6

This Stage2-only experiment addresses the V5 safety-gate result where every
replacement profile lost complete correct components. V6 never replaces the
production Stage2 result. It runs accepted production Stage2 first, then appends
only novel residual lanes that are:

- actual occupied voxels labelled line by the deployed calibrated V4 Stage1 decision;
- accepted by the production Stage2 refiner or the existing strict geometry override;
- materially novel relative to production-accepted line voxels;
- parallel to and longitudinally overlapping a production-accepted sibling line;
- within a bounded cross-section offset;
- free of synthetic voxels and runtime GT use.

GT is used only by the offline selector. The final runtime processor never reads
raw_labels. The canonical output layout remains:

    .../v4_stage23_quality/full_run_v2_<UTC>/
        stage2/<SID>/
        stage3/              # reserved, empty
        selection/
        logs/
        timing/
        status/

Terminal selector outcomes:

- SELECTED_SAFE_RESIDUAL_UNION_PROFILE: run Stage2 for requested sessions.
- NO_SAFE_RESIDUAL_UNION_PROFILE: package diagnostics and exit 3 without an
  all-session Stage2 run.

Primary files:

- v4_stage2_residual_union.py
- select_v4_stage2_residual_union_profile.py
- run_v4_stage2_residual_union.py
- self_test_residual_union.py
- run_v4_stage2_residual_union_v6_on_nebius.sh
- monitor_v4_stage2_residual_union_v6_on_nebius.sh
- download_v4_stage2_residual_union_v6_to_mac.sh
