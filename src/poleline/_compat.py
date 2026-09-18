"""Compatibility for historical module names and location-sensitive entry points.

This adapter changes file discovery only. The migrated implementations retain
their original computation and CLI parsing. Use an editable checkout.
"""
from pathlib import Path
import sys
import json
import importlib
import importlib.abc
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
_MOVES = json.loads((ROOT / 'docs/layout-map.json').read_text())['moves']


def configure(namespace, old_path, canonical_name=None):
    old = ROOT / old_path
    namespace['__file__'] = str(old)
    # Preserve the old entry point's import precedence, including the distinct
    # V4 and V6 voxel_common/precision_common implementations.
    parent = str(old.parent)
    if old_path.startswith(('v4/', 'ops/')) and str(ROOT / 'v4') not in sys.path:
        sys.path.insert(0, str(ROOT / 'v4'))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    name = namespace.get('__name__')
    legacy_name = old.stem
    if canonical_name and name in (canonical_name, legacy_name):
        module = sys.modules.get(name)
        if module is not None:
            parent_name, _, leaf = canonical_name.rpartition('.')
            parent_module = importlib.import_module(parent_name)
            sys.modules.setdefault(canonical_name, module)
            sys.modules.setdefault(legacy_name, module)
            setattr(parent_module, leaf, sys.modules[canonical_name])


def execute(namespace, target):
    path = ROOT / target
    for move in _MOVES:
        if move['new'] == target and move['kind'] == 'python':
            canonical = target[4:-3].replace('/', '.') if target.startswith('src/') else None
            configure(namespace, move['old'], canonical)
            break
    code = compile(path.read_bytes(), str(path), 'exec')
    exec(code, namespace)


def implementation_path(old_path):
    """Resolve source-inspection checks to implementation, not adapter text."""
    path = Path(old_path)
    relative = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    for entry in _MOVES:
        if entry['old'] == relative:
            return ROOT / entry['new']
    return ROOT / relative


class _CanonicalLoader(importlib.abc.Loader):
    def __init__(self, move):
        self.move = move

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        execute(module.__dict__, self.move['new'])


class _CanonicalFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for move in _MOVES:
            if move['kind'] == 'python' and move['new'].startswith('src/'):
                name = move['new'][4:-3].replace('/', '.')
                if name == fullname:
                    return importlib.util.spec_from_file_location(
                        fullname, ROOT / move['new'], loader=_CanonicalLoader(move))
        return None


def install_import_compatibility():
    if not any(isinstance(finder, _CanonicalFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _CanonicalFinder())
