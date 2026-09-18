# macOS local repository and GitHub setup

Target local directory:

```text
~/dev/Pole_Line_3D_Recon
```

Target GitHub repository:

```text
https://github.com/agnidasgupta/Pole_Line_3D_Recon
```

## Create the local tree from the provided package

```bash
mkdir -p ~/dev
cd ~/dev
unzip ~/Downloads/Pole_Line_3D_Recon_GitHub.zip
cd Pole_Line_3D_Recon
```

## Validate before committing

```bash
python3 -m compileall -q .
for f in *.sh; do bash -n "$f"; done

git diff --no-index /dev/null .gitignore >/dev/null 2>&1 || true
find . -maxdepth 2 -type f | sort
```

Check that model/cache files are absent:

```bash
find . -type f \
  \( -name '*.npz' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \
     -o -name '*.joblib' -o -name '*.pkl' -o -name '*.onnx' \
     -o -name '*.engine' -o -name '*.safetensors' -o -name '*.tflite' \) \
  -print
```

The command should print nothing.

## Initialize Git

```bash
git init
git branch -M main
git add .
git status
git diff --cached --stat
git commit -m "Initial V6.2 three-stage pole-line reconstruction pipeline"
```

## Create and push with GitHub CLI

Install GitHub CLI if required:

```bash
brew install gh
```

Authenticate:

```bash
gh auth login
```

Create a private repository and push:

```bash
gh repo create agnidasgupta/Pole_Line_3D_Recon \
  --private \
  --source=. \
  --remote=origin \
  --push
```

Use `--public` instead of `--private` only if the code is intended to be publicly visible.

Verify:

```bash
git remote -v
git status
gh repo view agnidasgupta/Pole_Line_3D_Recon --web
```

## If the GitHub repository was created manually in the browser instead

Create an empty repository named `Pole_Line_3D_Recon` under `agnidasgupta`, with no generated README/gitignore/license, then run:

```bash
git remote add origin git@github.com:agnidasgupta/Pole_Line_3D_Recon.git
git push -u origin main
```

If HTTPS is preferred:

```bash
git remote add origin https://github.com/agnidasgupta/Pole_Line_3D_Recon.git
git push -u origin main
```
