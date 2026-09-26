# E10 Stage1-to-Stage2 in-memory handoff result

**Update date: 2026-09-25 UTC**
**Decision: NOT_ACCEPTED**

## Numeric summary

| Metric | Disk control | In-memory candidate |
| --- | ---: | ---: |
| Sessions | 30 | 29 |
| Timed slices (first slice/session excluded) | 3708 | 3708 |
| Combined mean (ms) | 2298.651 | 2326.666 |
| Combined P50 (ms) | 990.382 | 989.655 |
| Combined P95 (ms) | 8806.627 | 9027.193 |
| Mean time saved | — | -1.219% |
| Stage1 exact payload rows | — | 3738/3738 |
| Stage2 output comparisons | — | 32/32 |
| Complete slice coverage | — | PASS |
| Complete timing rows | — | True |

| Sessions faster | — | 13/29 |
| Candidate P95 <= control P95 + 5% | — | True |

The candidate timing excludes the exactness diagnostic NPZ read but includes its Stage1 manifest write. Control and candidate both write V10 Stage2 outputs; only the control writes and reloads Stage1 NPZ/JSON. The control Stage2 wall and candidate callback measure analogous but not perfectly identical boundaries. A 2% mean gain, faster majority of sessions, and P95 within 5% are required before acceptance.

## Control timing breakdown

- `csv_read_ms`: count=3708, mean=73.6437046815003, P50=64.69702200013216, P95=147.98792109986596
- `sparse_item_prep_ms`: count=3708, mean=22.692236473835187, P50=18.50989700028549, P95=49.83318834988494
- `stage1_wall_ms`: count=3708, mean=105.03984751968954, P50=98.76308700017944, P95=155.06895690039062
- `stage1_artifact_write_ms`: count=3708, mean=9.860742549353583, P50=8.893496500149922, P95=16.844066299836413
- `stage1_manifest_write_ms`: count=3708, mean=9.146093389430085, P50=6.502653499865119, P95=23.650765550291904
- `stage1_load_ms`: count=3708, mean=4.56601003776565, P50=4.08324099953461, P95=8.295756299958153
- `production_stage2_ms`: count=3708, mean=100.02095785518269, P50=98.57783099960216, P95=167.33410719907621
- `stage1_electrical_track_ms`: count=3708, mean=1922.55713527183, P50=629.0197739999712, P95=8244.981650849872
- `stage2_total_ms`: count=3708, mean=2026.7363938263322, P50=731.268686000476, P95=8435.57200379977
- `stage2_disk_slice_wall_ms`: count=3708, mean=2078.267998589797, P50=768.889954500537, P95=8539.842102450528
- `combined_control_ms`: count=3708, mean=2298.650623203606, P50=990.3824814991822, P95=8806.626882000137

## Candidate timing breakdown

- `csv_read_ms`: count=3708, mean=67.39888510813235, P50=59.87688949880976, P95=133.2526276997669
- `sparse_item_prep_ms`: count=3708, mean=22.26587300624865, P50=18.05177200003527, P95=49.12445510062751
- `stage1_inference_ms`: count=3708, mean=105.18843991479596, P50=98.76968499975192, P95=155.35791635020357
- `stage1_manifest_write_ms`: count=3708, mean=7.433614711439844, P50=5.496856999343436, P95=18.73685144764749
- `stage2_reconstruction_ms`: count=3708, mean=2067.238092151019, P50=745.5135114996665, P95=8603.795759850975
- `stage2_write_and_orchestration_ms`: count=3708, mean=57.14090516613899, P50=41.856422500131885, P95=133.38684784794168
- `stage2_callback_and_output_ms`: count=3708, mean=2124.3789973171574, P50=787.6513834999059, P95=8703.188231099914
- `combined_stage1_stage2_ms`: count=3708, mean=2326.6658100577747, P50=989.6553650014538, P95=9027.193068598805
- `validation_ms_excluded`: count=3708, mean=8.798110624618507, P50=7.343713999944157, P95=18.304107101994283

## Raw numbers

```json
{
  "acceptance_rules": {
    "complete_slice_coverage": true,
    "expected_comparisons": 32,
    "expected_sessions": 30,
    "expected_slices": 3738,
    "max_p95_increase_percent": 5.0,
    "minimum_faster_sessions": 16,
    "minimum_saved_percent": 2.0,
    "stage1_exact_rows": 3738,
    "stage2_output_comparisons_pass": true
  },
  "candidate_breakdown": {
    "combined_stage1_stage2_ms": {
      "count": 3708,
      "mean_ms": 2326.6658100577747,
      "p50_ms": 989.6553650014538,
      "p95_ms": 9027.193068598805
    },
    "csv_read_ms": {
      "count": 3708,
      "mean_ms": 67.39888510813235,
      "p50_ms": 59.87688949880976,
      "p95_ms": 133.2526276997669
    },
    "sparse_item_prep_ms": {
      "count": 3708,
      "mean_ms": 22.26587300624865,
      "p50_ms": 18.05177200003527,
      "p95_ms": 49.12445510062751
    },
    "stage1_inference_ms": {
      "count": 3708,
      "mean_ms": 105.18843991479596,
      "p50_ms": 98.76968499975192,
      "p95_ms": 155.35791635020357
    },
    "stage1_manifest_write_ms": {
      "count": 3708,
      "mean_ms": 7.433614711439844,
      "p50_ms": 5.496856999343436,
      "p95_ms": 18.73685144764749
    },
    "stage2_callback_and_output_ms": {
      "count": 3708,
      "mean_ms": 2124.3789973171574,
      "p50_ms": 787.6513834999059,
      "p95_ms": 8703.188231099914
    },
    "stage2_reconstruction_ms": {
      "count": 3708,
      "mean_ms": 2067.238092151019,
      "p50_ms": 745.5135114996665,
      "p95_ms": 8603.795759850975
    },
    "stage2_write_and_orchestration_ms": {
      "count": 3708,
      "mean_ms": 57.14090516613899,
      "p50_ms": 41.856422500131885,
      "p95_ms": 133.38684784794168
    },
    "validation_ms_excluded": {
      "count": 3708,
      "mean_ms": 8.798110624618507,
      "p50_ms": 7.343713999944157,
      "p95_ms": 18.304107101994283
    }
  },
  "candidate_combined": {
    "count": 3708,
    "mean_ms": 2326.6658100577747,
    "p50_ms": 989.6553650014538,
    "p95_ms": 9027.193068598805
  },
  "control_breakdown": {
    "combined_control_ms": {
      "count": 3708,
      "mean_ms": 2298.650623203606,
      "p50_ms": 990.3824814991822,
      "p95_ms": 8806.626882000137
    },
    "csv_read_ms": {
      "count": 3708,
      "mean_ms": 73.6437046815003,
      "p50_ms": 64.69702200013216,
      "p95_ms": 147.98792109986596
    },
    "production_stage2_ms": {
      "count": 3708,
      "mean_ms": 100.02095785518269,
      "p50_ms": 98.57783099960216,
      "p95_ms": 167.33410719907621
    },
    "sparse_item_prep_ms": {
      "count": 3708,
      "mean_ms": 22.692236473835187,
      "p50_ms": 18.50989700028549,
      "p95_ms": 49.83318834988494
    },
    "stage1_artifact_write_ms": {
      "count": 3708,
      "mean_ms": 9.860742549353583,
      "p50_ms": 8.893496500149922,
      "p95_ms": 16.844066299836413
    },
    "stage1_electrical_track_ms": {
      "count": 3708,
      "mean_ms": 1922.55713527183,
      "p50_ms": 629.0197739999712,
      "p95_ms": 8244.981650849872
    },
    "stage1_load_ms": {
      "count": 3708,
      "mean_ms": 4.56601003776565,
      "p50_ms": 4.08324099953461,
      "p95_ms": 8.295756299958153
    },
    "stage1_manifest_write_ms": {
      "count": 3708,
      "mean_ms": 9.146093389430085,
      "p50_ms": 6.502653499865119,
      "p95_ms": 23.650765550291904
    },
    "stage1_wall_ms": {
      "count": 3708,
      "mean_ms": 105.03984751968954,
      "p50_ms": 98.76308700017944,
      "p95_ms": 155.06895690039062
    },
    "stage2_disk_slice_wall_ms": {
      "count": 3708,
      "mean_ms": 2078.267998589797,
      "p50_ms": 768.889954500537,
      "p95_ms": 8539.842102450528
    },
    "stage2_total_ms": {
      "count": 3708,
      "mean_ms": 2026.7363938263322,
      "p50_ms": 731.268686000476,
      "p95_ms": 8435.57200379977
    }
  },
  "control_combined": {
    "count": 3708,
    "mean_ms": 2298.650623203606,
    "p50_ms": 990.3824814991822,
    "p95_ms": 8806.626882000137
  },
  "decision": "NOT_ACCEPTED",
  "observed": {
    "candidate_slices": 3738,
    "complete_slice_coverage": true,
    "control_slices": 3738,
    "faster_sessions": 13,
    "p95_non_regression": true,
    "saved_percent": -1.2187666351454656,
    "sessions": 30,
    "stage1_exact_rows": 3738,
    "stage2_comparisons": 32,
    "stage2_output_comparisons_pass": true,
    "timing_complete": true
  },
  "raw_sources": {
    "candidate_timings": "/outputs/v4_stage12_persistent_stage2_e10/20260925T135739Z/candidate/e10_persistent_full30/timings/stage12",
    "comparison_dir": "/outputs/v4_stage12_persistent_stage2_e10/20260925T135739Z/comparisons",
    "control_stage1_timings": "/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/20260925T141742Z_e10_control_full30/timings/stage1",
    "control_stage2_timings": "/outputs/v4_stage12_persistent_stage2_e10/20260925T135739Z/control/e10_control_full30/stage2/timings"
  },
  "timing_scope": "sum of Stage1 CSV read, sparse prep, inference and Stage2 disk slice wall for control; analogous Stage1 components plus direct Stage2 callback for candidate; first slice of each session excluded; diagnostic production NPZ reload excluded in candidate",
  "update_date_utc": "2026-09-25"
}
```
