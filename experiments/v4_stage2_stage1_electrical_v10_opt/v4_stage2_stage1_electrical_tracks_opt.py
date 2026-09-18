#!/usr/bin/env python3
from pathlib import Path
import sys
_root=Path(__file__).absolute().parents[2]
if not (_root/"src/poleline").is_dir(): _root=Path("/workspace/poleline_repo")
sys.path.insert(0,str(_root/"src"))
from poleline._compat import execute
execute(globals(), 'src/poleline/stage2/reconstruction.py')
