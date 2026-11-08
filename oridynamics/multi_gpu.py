"""Run independent JSON targets on separate GPUs using the standard sampler."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from oridynamics import batch_predict
from oridynamics.complex_inputs import load_native_targets, native_sample
from oridynamics.prepare import anchor_sampling_spec
from oridynamics.output_paths import output_directory, timestamped_output

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = (
    'paths', 'assets_root', 'seeds', 'samples_per_seed', 'sample_chunk_size',
    'anchor_candidates', 'anchor_samples_per_seed', 'anchor_seed', 'attention_chunk_size', 'attention_backend', 'dtype',
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-json', type=Path, required=True,
                   help='PTX task array with precomputed MSAs; each complete target stays on one GPU')
    p.add_argument('--gpus', required=True,
                   help='Comma-separated GPU indices; relative to CUDA_VISIBLE_DEVICES when set')
    p.add_argument('--out', type=Path,
                   help='Output directory; auto-suffix if it exists; default: runs/multigpu_<host>_<timestamp>_<unique ID>')
    defaults = batch_predict.parser()
    for name in OPTIONS:
        default = defaults.get_default(name)
        kwargs = dict(default=default)
        if name in ('paths', 'assets_root'):
            kwargs['type'] = Path
        elif isinstance(default, int):
            kwargs['type'] = int
        if name == 'dtype':
            kwargs['choices'] = ('bf16', 'fp32')
        elif name == 'attention_backend':
            kwargs['choices'] = ('torch', 'cuequivariance')
        p.add_argument('--' + name.replace('_', '-'), **kwargs)
    p.add_argument('--plan', action='store_true',
                   help='Validate inputs and paths and display assignment without creating files or loading models')
    return p


def resolve_gpus(value, environ):
    parts = value.split(',')
    if not parts or any(not re.fullmatch(r'\d+', part.strip()) for part in parts):
        raise ValueError('--gpus must contain nonnegative integer indices')
    indices = [int(part) for part in parts]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate GPU indices')
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return [(str(index), str(index)) for index in indices]
    devices = [part.strip() for part in visible.split(',')]
    if any(not part or part == '-1' for part in devices) or max(indices) >= len(devices):
        raise ValueError('--gpus exceeds the GPUs exposed by CUDA_VISIBLE_DEVICES')
    selected = [devices[index] for index in indices]
    if len(set(selected)) != len(selected):
        raise ValueError('Selected GPU indices resolve to the same device')
    return list(zip(map(str, indices), selected))


def automatic_output():
    return timestamped_output(Path('runs') / 'multigpu')


def split_targets(targets, gpus):
    # 整个复合物保持完整；按蛋白残基数粗略分配负载，不承诺运行时间均衡。
    groups = [dict(gpu=gpu, cuda_device=device, targets=[], residues=0)
              for gpu, device in gpus[:len(targets)]]
    def size(target):
        return sum(len(item['proteinChain']['sequence']) * item['proteinChain'].get('count', 1)
                   for item in target['native_input']['sequences'] if 'proteinChain' in item)
    for target in sorted(targets, key=size, reverse=True):
        group = min(groups, key=lambda item: item['residues'])
        sample = native_sample(target)
        for key in ('initial_structure', 'structure_chains'):
            if key in target:
                sample[key] = target[key]
        group['targets'].append(sample)
        group['residues'] += size(target)
    return groups


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stop_workers(processes):
    # 只清理本启动器创建的独立进程组，避免遗留占卡进程。
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_workers(root, groups, options):
    requested_output = root.absolute()
    root = output_directory(requested_output, create=True)
    (root / 'inputs').mkdir()
    (root / 'logs').mkdir()
    report = dict(status='running', requested_output=str(requested_output), output=str(root), workers=[])
    processes = []
    report_path = root / 'multi_gpu.json'
    print(f'[Multi-GPU] Output: {root}', flush=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous_term = signal.signal(signal.SIGTERM, interrupt)
    try:
        with ExitStack() as stack:
            for group in groups:
                gpu = group['gpu']
                source = root / 'inputs' / f'gpu_{gpu}.json'
                output = root / f'gpu_{gpu}'
                log = root / 'logs' / f'gpu_{gpu}.log'
                write_json(source, group['targets'])
                command = [sys.executable, '-B', '-u', '-m', 'oridynamics', 'sample',
                           '--input-json', str(source), '--out', str(output),
                           '--device', 'cuda:0', *options]
                names = [target['name'] for target in group['targets']]
                worker = dict(gpu=gpu, cuda_device=group['cuda_device'], targets=names,
                              output=str(output), log=str(log), command=command, status='running')
                report['workers'].append(worker)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=group['cuda_device'])
                handle = stack.enter_context(log.open('x'))
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                worker['pid'] = process.pid
                write_json(report_path, report)
                print(f"[GPU {gpu}] {', '.join(names)} -> {log}", flush=True)
            pending = set(range(len(processes)))
            while pending:
                for index in list(pending):
                    code = processes[index].poll()
                    if code is None:
                        continue
                    worker = report['workers'][index]
                    worker['returncode'] = code
                    worker['status'] = 'completed' if code == 0 else 'failed'
                    pending.remove(index)
                    write_json(report_path, report)
                    print(f"[GPU {worker['gpu']}] {worker['status']} (exit {code})", flush=True)
                if pending:
                    time.sleep(0.2)
        report['status'] = 'completed' if all(p.returncode == 0 for p in processes) else 'failed'
        return 0 if report['status'] == 'completed' else 1
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error)
        raise
    finally:
        stop_workers(processes)
        for worker, process in zip(report['workers'], processes):
            worker['returncode'] = process.returncode
            if worker['status'] == 'running':
                worker['status'] = 'completed' if process.returncode == 0 else report['status']
        for worker in report['workers'][len(processes):]:
            worker.update(status='failed', returncode=None)
        write_json(report_path, report)
        signal.signal(signal.SIGTERM, previous_term)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        gpus = resolve_gpus(args.gpus, os.environ)
        requested_output = (args.out if args.out is not None else automatic_output()).absolute()
        root = output_directory(requested_output)
        args.input_json = args.input_json.resolve(strict=True)
        args.paths = args.paths.resolve(strict=True)
        args.assets_root = args.assets_root.resolve(strict=True)
        options = []
        for name in OPTIONS:
            options.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
        # 复用正式入口的只读检查，不加载Torch或权重，不产生推理目录。
        with redirect_stdout(io.StringIO()):
            batch_predict.main(['--input-json', str(args.input_json), '--out', str(root),
                                '--plan', *options])
        args.msa_local_db, args.single_sequence, args.limit = None, False, None
        groups = split_targets(load_native_targets(args.input_json, args), gpus)
        if args.plan:
            print(json.dumps(dict(requested_output=str(requested_output), output=str(root), workers=[
                dict(gpu=g['gpu'], cuda_device=g['cuda_device'],
                     targets=[t['name'] for t in g['targets']]) for g in groups],
                seeds=args.seeds, samples_per_seed=args.samples_per_seed,
                sample_chunk_size=args.sample_chunk_size, anchor_candidates=args.anchor_candidates,
                anchor_sampling=anchor_sampling_spec(args.anchor_seed, args.anchor_candidates, args.anchor_samples_per_seed),
                status='preflight_only', gpu_availability='not_checked'), indent=2))
            return 0
        return run_workers(requested_output, groups, options)
    except KeyboardInterrupt:
        print('Interrupted; worker processes stopped.', file=sys.stderr)
        return 130
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
