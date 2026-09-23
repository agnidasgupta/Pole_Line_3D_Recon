# V4 Stage-2 native line-recall V2 experiment

This experiment does not modify Stage 1, the learned Stage-2 refiner, or Stage 3.
It evaluates the native pre-component V4 line candidate controls.

Profiles:

- baseline: candidate 0.08, weak 0.04, competition 0.55
- recall_mid: candidate 0.04, weak 0.01, competition 0.35
- recall_high: candidate 0.025, weak 0.005, competition 0.20

Recommended first run:

```bash
RUN_SCOPE=velasco_sweep nohup ./run_v4_stage2_line_recall_v2_on_nebius.sh > "$HOME/v4_stage2_line_recall_v2_master.log" 2>&1 &
```

After visual selection, run every saved Stage-1 session with one profile:

```bash
RUN_SCOPE=all SELECTED_PROFILE=recall_mid nohup ./run_v4_stage2_line_recall_v2_on_nebius.sh > "$HOME/v4_stage2_line_recall_v2_master.log" 2>&1 &
```
