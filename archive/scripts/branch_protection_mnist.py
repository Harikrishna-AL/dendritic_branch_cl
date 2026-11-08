"""Three-arm local pilot: frozen BDH -> trainable max branches -> fixed readout."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bdh_cl.models.protected_branches import ProtectedBranches
from bdh_cl.models.bdh_wrapper import BDHWrapper
from bdh_cl.config import BDHSpec
from bdh_cl.data.mnist import build_pools
from bdh_cl.scripts.frozen_mnist import balanced_indices, extract
from bdh_cl.scripts.run_cl import pick_device


def train(model, x, y, lr, epochs, arm, seed, step_offset=0):
    gen = torch.Generator().manual_seed(seed)
    logs = []
    step = step_offset
    for epoch in range(epochs):
        for ix in torch.randperm(len(y), generator=gen).split(32):
            bx, by = x[ix], y[ix]
            pre, soma, winners = model.responses(bx)
            logits = soma @ model.readout * 5.
            loss = F.cross_entropy(logits, by)
            prob = logits.detach().softmax(-1)
            eligible = (prob.argmax(-1) == by) & (prob.max(-1).values >= .6) & (soma.detach().max(-1).values > 0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.weight], 5.)
            raw = -lr * model.weight.grad
            delta = model.project(raw) if arm != 'ordinary' else raw
            freedom = float(delta.norm() / raw.norm().clamp_min(1e-12))
            with torch.no_grad():
                # Actual update is projected: no momentum, Adam, or weight decay.
                coords = torch.einsum('nkdr,nkd->nkr', model.basis, delta)
                violation = float(coords.abs().max())
                model.weight.add_(delta)
                if arm != 'ordinary':
                    model.consolidate(bx, winners.detach(), eligible, soma.detach() > 0)
                events = model.maybe_recruit(float(loss.detach()), freedom, raw) if arm == 'protected_recruit' else []
            model.weight.grad = None
            logs.append({'step': step, 'loss': float(loss.detach()), 'freedom': freedom,
                         'protected_update_violation': violation, 'eligible': int(eligible.sum()),
                         'recruited_neurons': events})
            step += 1
    return logs


@torch.no_grad()
def evaluate(model, evalsets):
    seen = 2*len(evalsets)
    values = [float((model(x)[:, :seen].argmax(-1) == y).float().mean()) for x, y in evalsets]
    return {'per_pair': values, 'all_seen': float(np.mean(values)),
            'old': float(np.mean(values[:-1])) if len(values)>1 else None, 'new': values[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default='bdh_cl/results/local_branch_protection_seed0')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'bdh_cl/results/local_frozen_mnist_seed0'
    old = json.loads((original/'results.json').read_text())
    encoder = BDHWrapper(BDHSpec(**old['encoder_spec'])).to(pick_device(args.device))
    encoder.load_state_dict(torch.load(original/'frozen_encoder.pt', map_location='cpu', weights_only=False)['model'])
    encoder.eval().requires_grad_(False)
    pools = build_pools(ROOT/'data_cache/mnist', side=14, task_tags=False)
    ntrain, ntest, epochs = (64, 20, 2) if args.smoke else (1000, 200, 10)
    valix = balanced_indices(pools[0].train_labels, pools[0].classes, 200, rng)
    data = []
    t0 = time.perf_counter()
    for i, pool in enumerate(pools):
        ix = balanced_indices(pool.train_labels, pool.classes, ntrain, rng, valix if i==0 else ())
        ti = balanced_indices(pool.test_labels, pool.classes, ntest, rng)
        tx, ty = extract(encoder, pool, ix, next(encoder.parameters()).device, 8)
        ex, ey = extract(encoder, pool, ti, next(encoder.parameters()).device, 8, test=True)
        data.append((tx, ty, ex, ey))
        print(f'Extracted {pool.name}: {len(ty)} train / {len(ey)} test', flush=True)
        if i==0:
            vx, vy = extract(encoder, pool, valix, next(encoder.parameters()).device, 8)
    del encoder
    # The feature cache is a runner convenience for identical arm streams;
    # each learner sees only the current stage's training data.
    init = ProtectedBranches()
    val = {}
    for lr in [.1, .5, 1.]:
        trial = copy.deepcopy(init)
        train(trial, data[0][0], data[0][1], lr, epochs, 'ordinary', 0)
        val[lr] = float((trial(vx)[:, :2].argmax(-1) == vy).float().mean())
    lr = max(val, key=lambda a: (val[a], -a))
    print(f'First-pair validation: {val}; fixed lr={lr}', flush=True)
    report = {'seed':0, 'source_encoder': str(original/'frozen_encoder.pt'),
              'encoder_pretraining':'digits 0/1 only, 500 steps; frozen',
              'ntrain_per_pair':ntrain, 'ntest_per_pair':ntest, 'epochs':epochs,
              'neurons':128, 'branches':4, 'initial_active_branches':3, 'rank_cap':16,
              'basis_storage_bytes':init.basis.numel()*4, 'learning_rate':lr,
              'validation_scores':val, 'confidence_threshold':.6,
              'recruitment':{'loss_ema_threshold':1., 'freedom_threshold':.2,
                             'patience_steps':20, 'neurons_per_event':8},
              'readout':'fixed random 128x10; scale 5; training competes across all 10 classes',
              'arms':{}}
    for arm in ['ordinary', 'protected', 'protected_recruit']:
        model = copy.deepcopy(init)
        rows, logs, references = [], [], []
        evalsets = []
        prototypes = []
        offset = 0
        for i, (x, y, ex, ey) in enumerate(data):
            history = train(model, x, y, lr, epochs, arm, i, offset)
            offset += len(history)
            logs.extend(history)
            evalsets.append((ex, ey))
            metrics = evaluate(model, evalsets)
            with torch.no_grad():
                # Store current class means once; no refreshed old prototypes.
                _, h, _ = model.responses(x)
                h = F.normalize(h, dim=-1)
                prototypes.extend([h[y==c].mean(0) for c in (2*i,2*i+1)])
                P = F.normalize(torch.stack(prototypes), dim=-1)
                pacc = []
                for xx, yy in evalsets:
                    hh = F.normalize(model.responses(xx)[1], dim=-1)
                    pacc.append(float(((hh@P.T).argmax(-1)==yy).float().mean()))
                current = model.responses(ex[:64])
                references.append(tuple(t.clone() for t in current))
                drift = []
                for (xx, _), ref in zip(evalsets[:-1], references[:-1]):
                    pre, soma, winners = model.responses(xx[:64])
                    drift.append({'preactivation_relative_l2':float((pre-ref[0]).norm()/ref[0].norm().clamp_min(1e-12)),
                                  'soma_relative_l2':float((soma-ref[1]).norm()/ref[1].norm().clamp_min(1e-12)),
                                  'winner_switch_fraction':float((winners!=ref[2]).float().mean())})
                rank = model.ranks[model.enabled]
            row = {'stage':i, 'metrics':metrics, 'prototype_per_pair':pacc,
                   'prototype_all_seen':float(np.mean(pacc)), 'old_response_drift':drift,
                   'mean_protected_rank':float(rank.float().mean()),
                   'rank_full_fraction':float((rank==model.rank).float().mean()),
                   'recruited_neurons_total':model.recruited,
                   'mean_update_freedom':float(np.mean([h['freedom'] for h in history])),
                   'max_projection_violation':max(h['protected_update_violation'] for h in history)}
            rows.append(row)
            print(f'{arm} after {i}: accuracy={metrics["all_seen"]:.3f}, old={metrics["old"]}, new={metrics["new"]:.3f}, rank={row["mean_protected_rank"]:.1f}, freedom={row["mean_update_freedom"]:.3f}, recruited={model.recruited}', flush=True)
            report['arms'][arm] = {'rows':rows, 'training':logs}
            report['elapsed_seconds'] = time.perf_counter()-t0
            (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        torch.save(model.state_dict(), out/f'{arm}.pt')
    print(f'Completed in {time.perf_counter()-t0:.1f}s: {out}', flush=True)

if __name__ == '__main__':
    main()
