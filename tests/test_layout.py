"""Migration checks: original computation bytes, adapters and package identities."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_canonical_computation_unchanged(self):
        manifest = json.loads((ROOT/'docs/layout-map.json').read_text())
        for move in manifest['moves']:
            if move['kind'] != 'python' or not move['new'].startswith('src/'):
                continue
            data = (ROOT/move['new']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), move['sha256'], move['new'])

    def test_aliases_resolve(self):
        manifest = json.loads((ROOT/'docs/layout-map.json').read_text())
        for move in manifest['moves']:
            self.assertTrue((ROOT/move['old']).exists(), move['old'])
            self.assertTrue((ROOT/move['new']).exists(), move['new'])

    def test_model_module_identity_both_import_orders(self):
        for imports in [
            'import voxel_common; import poleline.stage1.model as model',
            'import poleline.stage1.model as model; import voxel_common',
        ]:
            code = f'import sys; sys.path[:0] = [{str(ROOT/"src")!r}, {str(ROOT/"v4")!r}]; {imports}; assert model.MultiHeadVoxelNet3D is voxel_common.MultiHeadVoxelNet3D'
            subprocess.run([sys.executable, '-c', code], check=True, cwd=ROOT)

    def test_shell_syntax(self):
        for folder in ('scripts', 'experiments', 'legacy', 'v4'):
            for path in (ROOT/folder).rglob('*.sh'):
                subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
