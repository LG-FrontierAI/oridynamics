"""多卡启动器CPU回归：输入路径、整体系分配、进程隔离与失败传播；不运行模型。"""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from oridynamic import multi_gpu
from oridynamic.complex_inputs import load_native_targets

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


class MultiGpuTests(unittest.TestCase):
    def test_visibility_and_duplicate_guards(self):
        self.assertEqual(multi_gpu.resolve_gpus('1,0', {}), [('1', '1'), ('0', '0')])
        self.assertEqual(multi_gpu.resolve_gpus('0,1', {'CUDA_VISIBLE_DEVICES': '3,GPU-test'}),
                         [('0', '3'), ('1', 'GPU-test')])
        for value, env in [('0,0', {}), ('0,', {}), ('-1', {}),
                           ('1', {'CUDA_VISIBLE_DEVICES': '4'}),
                           ('0', {'CUDA_VISIBLE_DEVICES': ''}),
                           ('0', {'CUDA_VISIBLE_DEVICES': '-1'})]:
            with self.subTest(value=value, env=env), self.assertRaises(ValueError):
                multi_gpu.resolve_gpus(value, env)

    def test_real_examples_preserve_complete_systems_and_msas(self):
        args = SimpleNamespace(msa_local_db=None, single_sequence=False, limit=None)
        source = EXAMPLES / 'five_cases_with_msa.json'
        targets = load_native_targets(source, args)
        groups = multi_gpu.split_targets(targets, [('0', '0'), ('1', '1'), ('2', '2'), ('3', '3')])
        samples = [sample for group in groups for sample in group['targets']]
        original = {row['name']: row for row in json.loads(source.read_text())}
        self.assertCountEqual([row['name'] for row in samples], list(original))
        for sample in samples:
            self.assertEqual(len(sample['sequences']), len(original[sample['name']]['sequences']))
            for old, new in zip(original[sample['name']]['sequences'], sample['sequences']):
                if 'proteinChain' not in new:
                    self.assertEqual(new, old)
                    continue
                old, new = old['proteinChain'], new['proteinChain']
                self.assertEqual(old['sequence'], new['sequence'])
                self.assertEqual(old['id'], new['id'])
                self.assertEqual(Path(new['unpairedMsaPath']),
                                 (source.parent / old['unpairedMsaPath']).resolve())

    def test_relocated_ligand_and_anchor_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in [('a.a3m', '>q\nACD\n'), ('lig.sdf', 'fixture'), ('anchor.cif', 'fixture')]:
                (root / name).write_text(data)
            source = root / 'input.json'
            source.write_text(json.dumps([dict(name='complex', initial_structure='anchor.cif',
                structure_chains={'A': 'X', 'B': 'Y'}, sequences=[
                    {'proteinChain': dict(id=['A'], sequence='ACD', unpairedMsaPath='a.a3m')},
                    {'ligand': dict(id=['B'], ligand='FILE_lig.sdf')}])]))
            args = SimpleNamespace(msa_local_db=None, single_sequence=False, limit=None)
            groups = multi_gpu.split_targets(load_native_targets(source, args), [('0', '0')])
            sample = groups[0]['targets'][0]
            self.assertEqual(sample['initial_structure'], str(root / 'anchor.cif'))
            self.assertEqual(sample['sequences'][1]['ligand']['ligand'], 'FILE_' + str(root / 'lig.sdf'))
            self.assertEqual(sample['structure_chains'], {'A': 'X', 'B': 'Y'})

    def test_plan_creates_no_outputs_and_no_torch_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / 'weights.json'
            weights.write_text(json.dumps({'checkpoints': {key: 'fixture.ckpt' for key in ('native', 'codec', 'joint')}}))
            (root / 'fixture.ckpt').write_bytes(b'path-check-only')
            (root / 'common').mkdir()
            for name in ('components.cif', 'components.cif.rdkit_mol.pkl'):
                (root / 'common' / name).write_bytes(b'path-check-only')
            argv = ['--input-json', str(EXAMPLES / 'five_cases_with_msa.json'),
                    '--gpus', '0,1', '--paths', str(weights), '--assets-root', str(root),
                    '--out', str(root / 'new_run'), '--sample-chunk-size', '2', '--plan']
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='2,3', PYTHONDONTWRITEBYTECODE='1')
            code = 'import sys; from oridynamic.multi_gpu import main; assert main(' + repr(argv) + ") == 0; assert 'torch' not in sys.modules"
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=multi_gpu.ROOT,
                                    env=environment, text=True, capture_output=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual([w['cuda_device'] for w in plan['workers']], ['2', '3'])
            self.assertEqual(plan['sample_chunk_size'], 2)
            self.assertFalse((root / 'new_run').exists())
            (root / 'new_run').mkdir()
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=multi_gpu.ROOT,
                                    env=environment, text=True, capture_output=True, check=True)
            repeated = json.loads(result.stdout)
            self.assertNotEqual(repeated['output'], str(root / 'new_run'))
            self.assertEqual(repeated['requested_output'], str(root / 'new_run'))
            self.assertFalse(Path(repeated['output']).exists())
            self.assertEqual(list((root / 'new_run').iterdir()), [])

    def test_processes_isolate_outputs_and_propagate_failure(self):
        real_popen = subprocess.Popen
        fake_code = '''import json, os, pathlib, sys, time
args = sys.argv[1:]
out = pathlib.Path(args[args.index('--out') + 1]); out.mkdir()
source = pathlib.Path(args[args.index('--input-json') + 1])
targets = json.loads(source.read_text())
(out / 'received.json').write_text(json.dumps(dict(device=os.environ['CUDA_VISIBLE_DEVICES'], args=args)))
time.sleep(0.1)
raise SystemExit(7 if targets[0]['name'] == 'fail' else 0)
'''
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'run'
                groups = [dict(gpu='0', cuda_device='2', targets=[dict(name='good')]),
                          dict(gpu='1', cuda_device='3', targets=[dict(name='fail' if fail else 'good2')])]
                options = ['--seeds', '0,1,2,3', '--samples-per-seed', '8', '--sample-chunk-size', '2']
                def spawn(command, **kwargs):
                    self.assertTrue(kwargs['start_new_session'])
                    return real_popen([sys.executable, '-B', '-c', fake_code, *command[5:]], **kwargs)
                with mock.patch.object(multi_gpu.subprocess, 'Popen', side_effect=spawn), redirect_stdout(io.StringIO()):
                    status = multi_gpu.run_workers(root, groups, options)
                self.assertEqual(status, 1 if fail else 0)
                report = json.loads((root / 'multi_gpu.json').read_text())
                self.assertEqual(report['status'], 'failed' if fail else 'completed')
                for gpu, expected in [('0', '2'), ('1', '3')]:
                    received = json.loads((root / f'gpu_{gpu}' / 'received.json').read_text())
                    self.assertEqual(received['device'], expected)
                    self.assertEqual(received['args'][-len(options):], options)
                    self.assertEqual(received['args'][received['args'].index('--device') + 1], 'cuda:0')
                original = (root / 'multi_gpu.json').read_bytes()
                with mock.patch.object(multi_gpu.subprocess, 'Popen', side_effect=spawn), redirect_stdout(io.StringIO()):
                    self.assertEqual(multi_gpu.run_workers(root, groups, options), status)
                self.assertEqual((root / 'multi_gpu.json').read_bytes(), original)
                new_root, = root.parent.glob('run_*')
                repeated = json.loads((new_root / 'multi_gpu.json').read_text())
                self.assertEqual(repeated['output'], str(new_root))
                self.assertEqual(repeated['requested_output'], str(root))
                for worker in repeated['workers']:
                    self.assertTrue(Path(worker['output']).is_relative_to(new_root))
                    self.assertTrue((Path(worker['output']) / 'received.json').exists())

    def test_automatic_outputs_are_unique(self):
        first, second = multi_gpu.automatic_output(), multi_gpu.automatic_output()
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, Path('runs'))

    def test_launch_failure_cleans_started_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            groups = [dict(gpu=str(i), cuda_device=str(i), targets=[dict(name=f'target{i}')])
                      for i in range(2)]
            process = mock.Mock(pid=12345, returncode=-15)
            with mock.patch.object(multi_gpu.subprocess, 'Popen', side_effect=[process, OSError('launch failed')]), \
                 mock.patch.object(multi_gpu, 'stop_workers') as cleanup, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, 'launch failed'):
                    multi_gpu.run_workers(root, groups, [])
            cleanup.assert_called_once_with([process])
            report = json.loads((root / 'multi_gpu.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual([w['status'] for w in report['workers']], ['failed', 'failed'])


if __name__ == '__main__':
    unittest.main()
