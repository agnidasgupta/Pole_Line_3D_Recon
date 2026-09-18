#!/usr/bin/env python3
"""Compatibility entry point; implementation: tests/v4/smoke_test_v4_realtime.py."""
from pathlib import Path as _Path
import sys as _sys
_root = _Path(__file__).absolute().parents[1]
if not (_root / "src" / "poleline").is_dir():
    _root = _Path("/workspace/poleline_repo")
_sys.path.insert(0, str(_root / "src"))
from poleline._compat import execute as _execute
_execute(globals(), 'tests/v4/smoke_test_v4_realtime.py')
