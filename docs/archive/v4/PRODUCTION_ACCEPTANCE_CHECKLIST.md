# V4 production acceptance checklist

## Environment and persistence

- Docker image builds successfully on the H100 VM.
- No V4 Nebius host launcher invokes host Python.
- Auto-discovered input/output/model/calibration/Stage-2/session values are correct in `run_context.env`.
- Session inventory is persisted before GPU work.
- Code backup exists.
- Deployment fingerprint includes code + Stage 1 checkpoint + calibration + Stage 2 bundle.

## Source/error validation

- all Python files compile in Docker;
- all library modules import in Docker;
- all CLI entry points parse;
- missing paths/manifests/timing files fail clearly;
- missing x/y/z columns fail clearly;
- NaN/out-of-grid rows are handled;
- duplicate voxel coordinates are deterministic;
- duplicate slice source records are rejected;
- missing slice sequence numbers are allowed;
- malformed manifests are never silently overwritten;
- relocated durable stage artifacts are usable;
- 9 sequence gaps x 50 ft = 450 ft is enforced.

## Stage 1

- `full_cpu` reference succeeds.
- selected optimized runtime passes score tolerance and has zero Stage 1 label mismatches.
- selected runtime also preserves Stage 2 topology, geometry, and refiner probabilities within configured tolerances.
- selected batch size independently passes equivalence.
- Stage 1 timing includes sparse H2D, GPU scatter, GPU patch extraction, feature assembly, model, GPU gather, D2H, artifact write, and wall time.

## Stage 2

- Stage 2 consumes only current-slice Stage 1 output.
- Stage 2 can execute from saved/relocated Stage 1 artifacts with Stage 1 no longer present at its original path.
- Stage 2 durable pole/line/vertex artifacts and manifest are complete.
- Stage 2 timing is present.

## Stage 3

- Stage 3 can execute from saved/relocated Stage 2 artifacts.
- Stage 3 can be regenerated a second time without Stage 1/2 execution.
- newest `S` uses only `[S-9,S]`.
- at most 10 observed slice centers are present.
- missing sequence numbers are allowed.
- no future slice is used.
- no slice older than `S-9` is used.
- fragment join, span-completion, hidden-pole, chain/attachment, output-write, wrapper, and total timings are present.

## End-to-end

- quick rolling replay verifies.
- full selected-session replay verifies.
- `REALTIME_REPLAY_VERIFICATION.json` has `ok=true`.
- true arrival-to-publish P50/P95 is reviewed.
- Stage 1/2/3 P50/P95 breakdown is reviewed.
- Stage 3 latency by rolling-window observed-center count is reviewed.
- production marker says `V4_PRODUCTION_ACCEPTANCE_OK` for the current deployment fingerprint.
- review package and SHA256 are downloaded to the Mac and uploaded for review before GitHub V4 is updated.
