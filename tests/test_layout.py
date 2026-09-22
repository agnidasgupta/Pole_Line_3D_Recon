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
            self.assertEqual(hashlib.sha256(data).hexdigest(), move.get('current_sha256', move['sha256']), move['new'])

    def test_optimized_source_provenance(self):
        manifest = json.loads((ROOT/'docs/optimization-map.json').read_text())
        for entry in manifest['files']:
            actual = hashlib.sha256((ROOT/entry['destination']).read_bytes()).hexdigest()
            self.assertEqual(actual, entry['sha256'], entry['destination'])
            if entry['destination'].endswith('.py'):
                self.assertEqual(actual, entry['source_sha256'], entry['destination'])

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

    def test_optimization_alias_identity(self):
        pairs = [
            ('v4_conv_layout', 'poleline.stage1.kernels.conv_layout'),
            ('v4_groupnorm_layout', 'poleline.stage1.kernels.groupnorm_layout'),
            ('v4_core_schedule', 'poleline.stage1.scheduling'),
            ('v4_input_pack', 'poleline.stage1.input_pack'),
            ('v4_input_prefetch', 'poleline.io.input_prefetch'),
            ('v4_output_writer', 'poleline.io.output_writer'),
        ]
        for reverse in (False, True):
            code = f'import sys, importlib; sys.path[:0] = [{str(ROOT/"src")!r}, {str(ROOT/"v4")!r}]\n'
            for old, canonical in pairs:
                first, second = (canonical, old) if reverse else (old, canonical)
                code += f'assert importlib.import_module({first!r}) is importlib.import_module({second!r})\n'
            subprocess.run([sys.executable, '-c', code], check=True, cwd=ROOT)


if __name__ == '__main__':
    unittest.main()
