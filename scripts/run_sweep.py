"""Run every training configuration behind the README tables.

Each (seed, order) run is its own process with a watchdog: on some macOS machines
long-lived CPU PyTorch processes occasionally hang inside the BLAS thread pool. A hung
run is killed (whole process group) and retried. Configurations are spread over
--workers parallel lanes; one configuration's runs stay in one lane, so no two
processes append to the same results file.

  python scripts/run_sweep.py --bench mnist --workers 2
  python scripts/run_sweep.py --bench cifar10 --workers 3
  python scripts/run_sweep.py --bench mnist --suite hat       # HAT baseline, lr x dropout grid
  python scripts/run_sweep.py --bench mnist --suite gates     # gate-rule ablation
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALPHAS = [3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
SEEDS = {'mnist': [101, 102, 103], 'cifar10': [201, 202, 203]}
HAT_LRS = [.01, .03, .1, .3, 1.]
GATE_RULES = ['error_trace', 'fixed_random', 'random_per_batch', 'most_active', 'task_assigned']


def configs(bench, suite):
    """Each config is a list of extra train.py arguments."""
    if suite == 'hat':
        return [['--model', 'hat', '--lr', str(lr), '--dropout', str(d), '--key-routing'] for lr in HAT_LRS for d in [0, 1]]
    if suite == 'gates':
        return [['--model', 'branch', '--gate-rule', r, '--key-routing'] for r in GATE_RULES]
    if bench == 'mnist':
        cfg = [('branch', a, 0) for a in ['1', '2', '3']] + [('plain', a, 0) for a in ['1', '2', '3']] + [('plain', '1', al) for al in ALPHAS]
    else:
        cfg = [('branch', a, 0) for a in ['conv_b4_sparse', 'conv_b4', 'conv_b1_wide']] + [('plain', 'cnn', al) for al in [0] + ALPHAS]
    return [['--model', m, '--arch', a, '--alpha', str(al)] for m, a, al in cfg]


def run_one(cmd, timeout, retries=3):
    for attempt in range(retries):
        proc = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            print(f'[sweep] timeout, retry {attempt + 1}: {" ".join(cmd[2:])}', flush=True)
    print(f'[sweep] GAVE UP: {" ".join(cmd[2:])}', flush=True); return -1


def lane(jobs, bench, timeout):
    for extra in jobs:
        for seed in SEEDS[bench]:
            for first in [0, 4]:
                run_one([sys.executable, 'scripts/train.py', '--bench', bench, *extra, '--seeds', str(seed), '--orders', str(first)], timeout)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bench', choices=list(SEEDS), required=True)
    p.add_argument('--suite', choices=['main', 'hat', 'gates'], default='main')
    p.add_argument('--workers', type=int, default=2); p.add_argument('--timeout', type=int, default=300)
    a = p.parse_args(); cfgs = configs(a.bench, a.suite)
    lanes = [cfgs[i::a.workers] for i in range(a.workers)]
    threads = [threading.Thread(target=lane, args=(jobs, a.bench, a.timeout)) for jobs in lanes]
    for t in threads: t.start()
    for t in threads: t.join()
    print('[sweep] done', a.bench, a.suite, flush=True)


if __name__ == '__main__':
    main()
