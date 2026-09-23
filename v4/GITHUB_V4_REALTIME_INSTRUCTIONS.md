# GitHub V4 update workflow — only after Nebius acceptance

Repository:

```text
https://github.com/agnidasgupta/Pole_Line_3D_Recon
```

Target branch: `v4`. The repository already stores this implementation under top-level `v4/`.

Do **not** update the branch until the Nebius review package has been examined and the exact deployment reports `V4_PRODUCTION_ACCEPTANCE_OK`.

On the Mac, use ordinary Git/shell commands only; no Python validation is required on the host.

```bash
cd ~/dev/Pole_Line_3D_Recon
git fetch origin
git switch v4
git pull --ff-only
```

Back up the existing repository V4 directory before replacement:

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar -C . -czf "$HOME/Downloads/Pole_Line_3D_Recon_v4_before_${STAMP}.tar.gz" v4
```

Extract the accepted code bundle to a temporary directory, locate it programmatically, and synchronize only the repository `v4/` directory:

```bash
TMP=$(mktemp -d)
unzip -q "$HOME/Downloads/Pole_Line_3D_Recon_V4_Production_Ready_Nebius_Test.zip" -d "$TMP"
SRC=$(find "$TMP" -type f -name run_v4_production_tests_on_nebius.sh -print | head -1)
SRC=$(dirname "$SRC")
rsync -av --delete --exclude='__pycache__/' --exclude='*.pyc' "$SRC/" ./v4/
rm -rf "$TMP"
```

Review before committing:

```bash
git status --short
git diff --stat -- v4
git diff -- v4
```

Commit the accepted production changes:

```bash
git add v4
git commit -m "Optimize V4 realtime production inference and reconstruction"
git push origin v4
```

Optionally tag the accepted deployment after confirming the pushed tree matches the reviewed version:

```bash
git tag -a v4.0.0-production -m "V4 production runtime accepted on Nebius H100"
git push origin v4.0.0-production
```

Never use `rsync --delete` against the repository root. Scope it to `./v4/` so older versions remain untouched.
