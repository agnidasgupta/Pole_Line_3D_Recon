# V4 Replay Verifier Hotfix — 2026-08-25

## Failure fixed

The completed realtime runner writes the current Stage-3 replay contract as:

`stage3_rolling_past_only=true`

The previous verifier incorrectly required the obsolete field:

`stage3_multi_slice_only=true`

Therefore a correct quick replay could finish and still be rejected during post-replay verification.

## Runtime impact

None. This hotfix does **not** modify Stage 1 inference, Stage 2 reconstruction, Stage 3 reconstruction, runtime selection, batch selection, model weights, calibration, or the Stage-2 refiner bundle.

## Files changed/added

- `verify_v4_realtime_replay.py` — validates the current rolling-past-only contract and accepts older replay metadata with a warning.
- `smoke_test_v4_replay_verifier_contract.py` — regression test for current, legacy, and invalid >10-center windows.
- `validate_v4_production_source_contract.py` — reports the current Stage-3 contract name.
- `v4_code_validation_inside_docker.sh` — executes the regression test.
- `resume_v4_acceptance_after_verifier_hotfix_on_nebius.sh` — guarded host-side recovery launcher; no host Python.
- `v4_recovery_after_verifier_hotfix_inside_docker.sh` — Docker-only recovery body.
- `show_v4_production_state_on_nebius.sh` and `package_v4_review_bundle_on_nebius.sh` — include recovery status/evidence.

## Preferred recovery

After installing this hotfix into the same V4 directory used for the failed run:

```bash
cd <auto-discovered-v4-dir>
chmod +x *.sh
./resume_v4_acceptance_after_verifier_hotfix_on_nebius.sh
```

The recovery launcher reuses the completed H100 gate/batch/independent-stage evidence **only** if all of the following are true:

1. the prior failure occurred in `verify_v4_realtime_replay.py`;
2. the Stage-1 model SHA256 is unchanged;
3. calibration SHA256 is unchanged;
4. Stage-2 bundle SHA256 is unchanged;
5. all Stage-1/2/3 production runtime files are byte-for-byte identical to the code backup created for the failed run;
6. all runtime-gate and batch-selection files are byte-for-byte identical.

If any identity check fails, recovery exits and requires the full acceptance suite.

The recovery then:

1. reruns all static/synthetic validation inside Docker;
2. re-verifies the already completed quick replay with the corrected verifier;
3. runs the full selected-session realtime replay;
4. verifies the full replay;
5. generates timing summaries;
6. promotes the runtime/batch gate only after the full replay passes;
7. writes a production-acceptance marker for the current deployment fingerprint.

All Python remains inside Docker.
