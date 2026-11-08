"""Matched objective ablation with boundary-free online context prototypes.

Numenta-inspired control, not a reproduction of its architecture or clustering.
Router observes features and contemporaneous gates, never labels or phase IDs.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from torch.nn import functional as F

from bdh_cl.data.mnist import SPLITS, load_raw
from bdh_cl.scripts.split_mnist_usefulness import make
from bdh_cl.scripts.branch_associative_retrieval import describe, normalize, responses, predict, metrics


class ContextBank:
    def __init__(self, dimension, slots=95):
        self.centers = np.zeros((slots, dimension), np.float32)
        self.counts = np.zeros(slots, np.int64)
        self.votes = np.zeros((slots, 24, 4), np.int64)
        self.used = 0

    def observe(self, features, route):
        # Membership uses input features only, with no route-conditioned search.
        for feature in features:
            best = int((normalize(self.centers[:self.used]) @ feature).argmax()) if self.used else 0
            novel = self.used == 0 or float(normalize(self.centers[best:best+1])[0] @ feature) < .85
            if novel and self.used < len(self.centers):
                best = self.used
                self.used += 1
            self.counts[best] += 1
            self.centers[best] += (feature - self.centers[best]) / self.counts[best]
            self.votes[best, np.arange(24), route] += 1

    def retrieve(self, features, centers=None):
        keys = normalize(self.centers[:self.used] if centers is None else centers)
        winners = (features @ keys.T).argmax(1)
        return self.votes[winners].argmax(-1)

    def size(self):
        return self.centers.nbytes + self.counts.nbytes + self.votes.nbytes


def digest(model, banks):
    h = hashlib.sha256()
    for v in model.state_dict().values():
        h.update(v.numpy().tobytes())
    for bank in banks.values():
        for v in (bank.centers, bank.counts, bank.votes):
            h.update(v.tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, banks, x, y, routes, seed):
    before = digest(model, banks)
    b = responses(model, x)
    features = {'input': describe(x.numpy()), 'latent': normalize(b.reshape(len(x), -1))}
    weight, bias = model.head.weight.numpy(), model.head.bias.numpy()
    current = np.broadcast_to(model.gates().argmax(-1).numpy(), (len(x), 24))
    rng = np.random.default_rng(seed)
    choices = {'current': current, 'oracle': routes[y // 2],
               'random_saved_route': routes[rng.integers(len(routes), size=len(x))]}
    for name, bank in banks.items():
        choices[name + '_prototype'] = bank.retrieve(features[name])
    scores = {name: metrics(predict(b, route, weight, bias), y, route, routes)
              for name, route in choices.items()}
    # Fixed perturbation of normalized prototype directions; no test-time updates.
    for name, bank in banks.items():
        keys = normalize(bank.centers[:bank.used])
        noise = normalize(np.random.default_rng(seed + 901).normal(size=keys.shape).astype('float32'))
        perturbed = keys + .1 * noise
        route = bank.retrieve(features[name], perturbed)
        scores[name + '_prototype_noise_0.1'] = metrics(predict(b, route, weight, bias), y, route, routes)
    subset = np.random.default_rng(20261031).choice(len(y), min(1000, len(y)), replace=False)
    separation = {}
    for name, z in features.items():
        separation[name] = {
            'class_silhouette_cosine': float(silhouette_score(z[subset], y[subset], metric='cosine')),
            'pair_silhouette_cosine': float(silhouette_score(z[subset], y[subset] // 2, metric='cosine')),
        }
    assert digest(model, banks) == before, 'Evaluation mutated learned state'
    return scores, separation


def report(out, records, protocol):
    lines = ['# Online prototype routing: matched auxiliary-objective ablation', '',
             'Three paired seeds; full Split MNIST, two epochs per pair, all ten output labels. '
             'Router receives no labels or phase boundaries; frozen evaluation on the official test set.', '',
             '| Objective | Router | Accuracy mean ± seed SD | Old digits 0–7 | Exact oracle-route agreement |',
             '|---|---|---:|---:|---:|']
    for objective in ['counterfactual', 'ce_only']:
        rows = [r for r in records if r['objective'] == objective]
        if not rows:
            continue
        for arm in rows[0]['scores']:
            values = [100*r['scores'][arm]['accuracy'] for r in rows]
            sd = statistics.stdev(values) if len(values) > 1 else 0
            old = statistics.mean(100*r['scores'][arm]['old_accuracy'] for r in rows)
            agreement = statistics.mean(100*r['scores'][arm]['exact_route_match'] for r in rows)
            lines.append(f'| {objective} | {arm} | {statistics.mean(values):.2f} ± {sd:.2f}% | {old:.2f}% | {agreement:.2f}% |')
    lines += ['', '| Objective | Learned-feature class silhouette | Learned-feature pair silhouette |', '|---|---:|---:|']
    for objective in ['counterfactual', 'ce_only']:
        rows = [r for r in records if r['objective'] == objective]
        if rows:
            vals = [statistics.mean(r['separation']['latent'][k] for r in rows)
                    for k in ['class_silhouette_cosine', 'pair_silhouette_cosine']]
            lines.append(f'| {objective} | {vals[0]:.4f} | {vals[1]:.4f} |')
    lines += ['', '## Protocol and interpretation', '',
              protocol['comparison'], '', protocol['router'], '', protocol['features'], '',
              'Oracle uses diagnostic end-of-phase routes and true pair identity, with the final weights and shared ten-class head. '
              'No output masking. Random chooses among the same saved routes independently of the image. '
              'These saved routes never enter prototype acquisition or autonomous inference.', '',
              'The router is a sidecar: training still uses the original error-derived global gate. '
              'Latent prototypes retain historical coordinate values while the encoder changes; no historical examples are stored or revisited. '
              'The fixed input-feature control removes encoder drift. Prototype votes can combine branches into a route never observed as a whole.', '',
              'Prototype-noise rows add a fixed isotropic direction perturbation of norm 0.1 to unit prototype directions. '
              'This is a sensitivity probe, not a comprehensive drift benchmark. Silhouette uses a fixed 1,000-test-image subset, '
              'with labels used for post-hoc diagnostics only. No model or hyperparameter is selected from test results.', '',
              'Limitations: Numenta-inspired online clustering, not an original-paper reproduction; no direct Numenta architecture comparison. '
              'One task order; reused seeds; fixed capacity and threshold; cumulative prototypes may saturate early. '
              'CE-only ablation retains error traces and their supervised feedback, isolating the auxiliary loss rather than removing the full method. '
              'No assertion of sparse executed compute, privacy guarantees, or general benchmark superiority.', '',
              'See results.json for per-seed results, acquisition drift, timing and memory; protocol.json for settings. '
              'Checkpoints and prototype banks are saved. Evaluation verifies unchanged weights and bank arrays.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', default='34,35,36')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    protocol = {**vars(args), 'epochs': 2, 'batch_size': 128, 'device': 'cpu',
                'comparison': 'Identical 24-neuron × 4-branch architecture, paired initialization/sample order, Adam .001, clip 5, EMA .95. Counterfactual objective: CE + .5 mean single-neuron-substitution CE. CE-only: CE, with identical counterfactual error-trace updates. Not all joint branch combinations.',
                'router': '95 online spherical prototypes; threshold .85; allocate on novelty while space remains, otherwise update nearest centroid by cumulative mean. Per-prototype neuron/branch vote counts associate observed pre-update gates. Membership independent of gates. Every current example observed before optimizer update; no labels, stage IDs, historical input buffer, boundary reset, or retrospective fitting.',
                'features': 'Input control: normalized 2×2 pooled pixels (196D). Learned-feature condition: normalized flattened 24×4 tanh branch outputs (96D), independent of the selected route. Same capacity and rule, different byte costs explicitly reported.',
                'scope': 'Numenta-inspired routing control, not exact reproduction. Full 60,000 training / 10,000 test images; fixed natural pair order; no tuning.'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    (out/'model_source.py').write_bytes(Path('bdh_cl/scripts/split_mnist_usefulness.py').read_bytes())
    raw = load_raw('data_cache/mnist')
    x = torch.tensor(raw['train_images'].copy(), dtype=torch.float32).flatten(1)/255
    y = torch.tensor(raw['train_labels'].copy(), dtype=torch.long)
    tx = torch.tensor(raw['test_images'].copy(), dtype=torch.float32).flatten(1)/255
    ty = raw['test_labels'].astype('int64')
    records = []
    for seed in map(int, args.seeds.split(',')):
        for objective, coefficient in [('counterfactual', .5), ('ce_only', 0.)]:
            torch.manual_seed(seed)
            model = make('local', 'hard')
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            banks = {'input': ContextBank(196), 'latent': ContextBank(96)}
            routes, acquired_means, stages = [], [], []
            started = time.perf_counter()
            for stage, pair in enumerate(SPLITS):
                ix = torch.nonzero((y==pair[0]) | (y==pair[1]), as_tuple=True)[0]
                for epoch in range(2):
                    gen = torch.Generator().manual_seed(seed*100000+stage*100+epoch)
                    for ids in ix[torch.randperm(len(ix), generator=gen)].split(128):
                        bx, by = x[ids], y[ids]
                        route = model.gates().argmax(-1).numpy()
                        with torch.no_grad():
                            latent = normalize(torch.tanh(model.first(bx)).numpy())
                        banks['input'].observe(describe(bx.numpy()), route)
                        banks['latent'].observe(latent, route)
                        opt.zero_grad(set_to_none=True)
                        logits, candidates = model(bx, counterfactual=True)
                        targets = by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
                        errors = F.cross_entropy(candidates.reshape(-1,10), targets, reduction='none').reshape(candidates.shape[:-1])
                        loss = F.cross_entropy(logits, by) + coefficient*errors.mean()
                        feedback = errors.detach().mean(0)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        opt.step()
                        model.observe(feedback)
                routes.append(model.gates().argmax(-1).numpy().copy())
                with torch.no_grad():
                    z = normalize(responses(model, tx).reshape(len(tx), -1))
                acquired_means.append(np.stack([z[ty==c].mean(0) for c in pair]))
                drift = [float(np.linalg.norm(np.stack([z[ty==c].mean(0) for c in SPLITS[k]])-acquired_means[k], axis=1).mean()) for k in range(stage)]
                stages.append({'stage': stage+1, 'occupied': {k:b.used for k,b in banks.items()}, 'old_class_centroid_l2_drift': drift})
                print(seed, objective, 'stage', stage+1, stages[-1], flush=True)
            training_seconds = time.perf_counter()-started
            reference_match = None
            if objective == 'counterfactual':
                reference = Path(f'results/split_mnist_compute_comparison_20261013/{seed}_local.pt')
                if reference.exists():
                    state = torch.load(reference, weights_only=True)
                    reference_match = all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
                    assert reference_match
            scores, separation = evaluate(model, banks, tx, ty, np.stack(routes), seed)
            record = {'seed':seed, 'objective':objective, 'scores':scores, 'separation':separation,
                      'stages':stages, 'training_and_router_seconds':training_seconds,
                      'router_numeric_bytes':{k:b.size() for k,b in banks.items()}, 'original_checkpoint_exact_match':reference_match}
            records.append(record)
            torch.save(model.state_dict(), out/f'{seed}_{objective}.pt')
            np.savez(out/f'{seed}_{objective}_banks.npz', routes=np.stack(routes), **{f'{k}_{a}':getattr(b,a) for k,b in banks.items() for a in ['centers','counts','votes']})
            (out/'results.json').write_text(json.dumps(records, indent=2))
            report(out, records, protocol)
            print(seed, objective, {k:round(v['accuracy']*100,2) for k,v in scores.items()}, flush=True)
    print('Completed:',out,flush=True)


if __name__ == '__main__':
    main()
