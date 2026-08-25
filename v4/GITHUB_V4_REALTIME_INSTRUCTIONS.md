# GitHub workflow — V4 realtime production candidate

Use the direct ZIP on Nebius first. Push the candidate only after the H100 runtime gate, Stage-2 training, full-session realtime replay, and rolling Stage-3 verification have been reviewed.

Repository target:

```text
https://github.com/agnidasgupta/Pole_Line_3D_Recon
```

Recommended validation branch:

```text
v4-realtime-production-candidate
```

## 1. Put the candidate under `~/dev` on the Mac

```bash
mkdir -p ~/dev
cd ~/dev
rm -rf Pole_Line_3D_Recon_V4_Realtime_Production_Candidate

unzip \
  ~/Downloads/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate.zip

cd ~/dev/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate
```

Validate syntax locally:

```bash
python3 -m compileall -q .

for f in *.sh; do
  bash -n "$f" || exit 1
done

find . -type d -name '__pycache__' -prune -exec rm -rf {} +
```

## 2. After Nebius validation, create/update the Git branch

Assuming the existing canonical clone is:

```text
~/dev/Pole_Line_3D_Recon
```

run:

```bash
cd ~/dev/Pole_Line_3D_Recon

git status
git fetch origin
```

If the validation branch does not yet exist:

```bash
git switch -c v4-realtime-production-candidate
```

If it already exists:

```bash
git switch v4-realtime-production-candidate
git pull --ff-only
```

Copy the accepted candidate into the repository while preserving `.git`:

```bash
rsync -av \
  --delete \
  --exclude='.git/' \
  ~/dev/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate/ \
  ~/dev/Pole_Line_3D_Recon/
```

Run code checks again:

```bash
cd ~/dev/Pole_Line_3D_Recon

python3 -m compileall -q .
for f in *.sh; do bash -n "$f" || exit 1; done
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
```

Review exactly what changed:

```bash
git status
git diff --stat
git diff
```

Commit only after review:

```bash
git add .

git commit -m "Add V4 realtime production candidate pipeline"

git push -u origin v4-realtime-production-candidate
```

## 3. Deploy the Git branch to Nebius

For a new clone:

```bash
cd /workspace/voxel_poleline

git clone \
  --branch v4-realtime-production-candidate \
  git@github.com:agnidasgupta/Pole_Line_3D_Recon.git \
  Pole_Line_3D_Recon_V4_Realtime_Production_Candidate

cd Pole_Line_3D_Recon_V4_Realtime_Production_Candidate
chmod +x *.sh *.py
```

For an existing validation clone:

```bash
cd /workspace/voxel_poleline/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate

git fetch origin
git switch v4-realtime-production-candidate
git pull --ff-only
```

Then run:

```bash
./build_v4_realtime_image_on_nebius.sh
./run_v4_realtime_preflight_on_nebius.sh
```

Continue with `NEBIUS_V4_REALTIME_RUNBOOK.md`.

## 4. Promotion to `main`

Do not merge the branch into `main` until all of these are accepted:

```text
Stage-1 H100 runtime equivalence gate
Stage-1 batch-size equivalence/speed sweep
Stage-2 local refiner metrics and thresholds
5-slice realtime replay verification
full-session realtime P50/P95 timing
strict rolling Stage-3 past-only verification
reconstruction output/audit review
```

After acceptance:

```bash
cd ~/dev/Pole_Line_3D_Recon

git switch main
git pull --ff-only
git merge --no-ff v4-realtime-production-candidate
git push origin main
```
