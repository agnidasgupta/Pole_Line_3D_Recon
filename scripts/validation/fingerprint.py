"""Fingerprint canonical code plus compatibility adapters, not generated output."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
paths = []
for folder in ('src', 'v4', 'scripts', 'experiments', 'legacy'):
    for p in (ROOT / folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and (
            p.suffix in ('.py', '.sh', '.cs', '.toml') or
            p.name.startswith(('requirements', 'Dockerfile'))
        ):
            paths.append(p)
paths.extend([ROOT/'pyproject.toml', ROOT/'docs/layout-map.json'])
digest = hashlib.sha256()
for p in sorted(set(paths)):
    digest.update(str(p.relative_to(ROOT)).encode() + b'\0')
    digest.update(hashlib.sha256(p.read_bytes()).digest())
print(digest.hexdigest())
