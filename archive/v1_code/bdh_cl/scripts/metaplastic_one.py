"""Run one metaplastic_control configuration in a fresh process and append it to the results JSON.
(Long-lived processes intermittently hang inside the BLAS thread pool on this machine.)"""
import json, os, sys, threading, time, numpy as np, torch
threading.Timer(240, lambda: os._exit(3)).start()  # self-kill: outer `timeout` cannot reach the framework child process
from pathlib import Path
from bdh_cl.scripts import metaplastic_control as mc
bench, arm, alpha, seed, first = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
torch.set_num_threads(2)
from bdh_cl.data import cifar10
d = cifar10.load('data_cache/cifar10')
data = [torch.from_numpy(np.array(v, copy=True)) for v in [d['images'][d['train']], d['labels'][d['train']], d['images'][d['validation']], d['labels'][d['validation']], d['test_images'], d['test_labels']]]
data = [v.long() if i % 2 else v for i, v in enumerate(data)]
t0 = time.perf_counter(); r = mc.run(bench, arm, alpha, seed, first, data)
path = Path(f'results/metaplastic_control_20261107/{bench}.json'); runs = json.loads(path.read_text()); runs.append(r); path.write_text(json.dumps(runs, indent=2))
print(bench, arm, alpha, seed, first, 'CIL', round(100 * r['test']['plain']['cil'], 2), 'TIL', round(100 * r['test']['plain']['til'], 2), 'loss', round(100 * r['til_retention_loss_val'], 1), 'frozen', r['frozen_frac_end'], f'{time.perf_counter() - t0:.0f}s', flush=True)
os._exit(0)
