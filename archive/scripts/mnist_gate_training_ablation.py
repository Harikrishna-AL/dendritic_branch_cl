"""Single-seed paired-order pilot isolating training-time branch selection."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier
from bdh_cl.scripts.matched_numenta_comparison import ROOT, predictions, metrics

ARMS = ['fixed_random', 'random_per_batch', 'error_trace', 'task_assigned', 'activation_wta']

class RoutedBranch(BranchClassifier):
    def __init__(self, arm):
        super().__init__('local', neurons=168, branches=4, gate='hard')
        self.selection_arm = arm
        self.register_buffer('selected', torch.zeros(168, dtype=torch.long))

    def gates(self):
        if self.selection_arm == 'error_trace':
            return super().gates()
        return F.one_hot(self.selected, 4).to(self.first.weight.dtype)


def digest(model):
    h = hashlib.sha256()
    for v in model.state_dict().values():
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(arms=args.arms, seed=args.seed, orders=['forward', 'reverse'], epochs_per_task=2,
                    batch_size=128, learning_rate=.001, neurons=168, branches=4,
                    auxiliary_loss=False, optimizer='continuous Adam, clip 5',
                    limitations=['One seed, not a significance study.',
                    'Fixed LR from earlier CE-only validation, not separately tuned per ablation.',
                    'Task-assigned means five fixed random configurations, not five disjoint branches.',
                    'Random-per-batch archive is the last sampled configuration, not its optimal route.',
                    'Activation WTA selects argmax mean absolute tanh activation per neuron per minibatch before the update. Its archive is the final selected minibatch configuration, not an optimized task route.',
                    'Input task means and archives use known training boundaries.',
                    'No weight snapshots, replay, pretrained features, or output label masking.'])
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    raw = load_raw(str(ROOT/'data_cache/mnist'))
    x = (torch.tensor(raw['train_images'].copy(), dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    y = torch.tensor(raw['train_labels'].copy(), dtype=torch.long)
    tx = (torch.tensor(raw['test_images'].copy(), dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    ty = torch.tensor(raw['test_labels'].copy(), dtype=torch.long)
    rng = np.random.default_rng(20261101)
    vid = np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==c))[:500] for c in range(10)])
    tid = np.setdiff1d(np.arange(len(y)), vid)
    assert not np.intersect1d(tid, vid).size
    np.savez(out/'split_indices.npz', train=tid, validation=vid)
    vx, vy = x[vid], y[vid]
    x, y = x[tid], y[tid]
    ix = [torch.where(y//2==k)[0] for k in range(5)]
    contexts = torch.stack([x[z].mean(0) for z in ix])
    torch.save(contexts, out/'contexts.pt')
    config_gen = torch.Generator().manual_seed(args.seed+820000)
    configs = torch.randint(4, (5,168), generator=config_gen)
    records = []
    for order_name, order in [('forward',list(range(5))), ('reverse',list(range(4,-1,-1)))]:
        for arm in args.arms:
            torch.manual_seed(args.seed)
            model = RoutedBranch(arm)
            initial_weights = hashlib.sha256(model.first.weight.detach().numpy().tobytes()).hexdigest()
            model.selected.copy_(configs[0])
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            routes = torch.zeros((5,168),dtype=torch.uint8)
            stages = []
            gate_counts = torch.zeros(168,4,dtype=torch.long)
            steps = presentations = 0
            trace_initial = model.error.clone()
            begin = time.perf_counter()
            for stage, group in enumerate(order):
                if arm=='task_assigned':
                    model.selected.copy_(configs[group])
                for epoch in range(2):
                    gen = torch.Generator().manual_seed(args.seed*100000+stage*100+epoch)
                    for ids in ix[group][torch.randperm(len(ix[group]), generator=gen)].split(128):
                        if arm=='random_per_batch':
                            gate_gen = torch.Generator().manual_seed(args.seed*1000000+steps)
                            model.selected.copy_(torch.randint(4,(168,),generator=gate_gen))
                        if arm=='activation_wta':
                            with torch.no_grad():
                                activations = torch.tanh(model.first(x[ids]).reshape(-1,168,4))
                                model.selected.copy_(activations.abs().mean(0).argmax(-1))
                        gate_counts += model.gates().long()
                        bx, by = x[ids], y[ids]
                        feedback = None
                        if arm=='error_trace':
                            with torch.no_grad():
                                _, candidates = model(bx, counterfactual=True)
                                targets = by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
                                errors = F.cross_entropy(candidates.reshape(-1,10),targets,reduction='none').reshape(candidates.shape[:-1])
                                feedback = errors.mean(0)
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(bx),by)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                        opt.step()
                        if feedback is not None:
                            model.observe(feedback)
                        steps += 1
                        presentations += len(ids)
                routes[group] = model.gates().argmax(-1).to(torch.uint8)
                mask = vy//2==group
                pred = predictions(model,'branch_ce',vx[mask],contexts,routes,'oracle',vy[mask]//2)
                stages.append(dict(group=group, acquisition_validation=float((pred==vy[mask]).float().mean())))
            seconds = time.perf_counter()-begin
            before = digest(model)
            route_before = routes.clone()
            scores = {mode: metrics(predictions(model,'branch_ce',tx,contexts,routes,mode,ty//2),ty,order)
                      for mode in ['current','inferred','oracle']}
            # Label-informed best-of-saved-routes diagnostic, not a deployable router.
            bank_preds = torch.stack([predictions(model,'branch_ce',tx,contexts,routes,'oracle',torch.full_like(ty,g)) for g in range(5)])
            scores['bank_any_correct'] = float((bank_preds==ty[None,:]).any(0).float().mean())
            final_val = predictions(model,'branch_ce',vx,contexts,routes,'oracle',vy//2)
            for st in stages:
                mask = vy//2==st['group']
                st['final_validation_oracle'] = float((final_val[mask]==vy[mask]).float().mean())
            assert digest(model)==before and torch.equal(routes,route_before)
            assert presentations==110000
            if arm!='error_trace':
                assert torch.equal(model.error,trace_initial)
            if arm=='fixed_random':
                assert torch.unique(routes,dim=0).shape[0]==1
            if arm=='task_assigned':
                assert torch.equal(routes.long(),configs)
            if arm=='random_per_batch':
                assert (gate_counts>0).all()
            row=dict(arm=arm,order=order_name,seed=args.seed,scores=scores,stages=stages,
                     train_seconds=seconds,presentations=presentations,steps=steps,
                     parameters=sum(p.numel() for p in model.parameters()),
                     unique_archived_routes=int(torch.unique(routes,dim=0).shape[0]),
                     fraction_neurons_visiting_all_branches=float((gate_counts>0).all(1).float().mean()),
                     initial_weight_hash=initial_weights)
            records.append(row)
            torch.save(dict(model=model.state_dict(),routes=routes,gate_counts=gate_counts,configs=configs),out/f'{args.seed}_{order_name}_{arm}.pt')
            (out/'results.json').write_text(json.dumps(records,indent=2))
            print(order_name,arm,{k:round(100*v['accuracy'],2) for k,v in scores.items() if isinstance(v,dict)},flush=True)
    assert len({r['initial_weight_hash'] for r in records})==1
    lines=['# MNIST training-gate ablation pilot','','One seed (101), two paired orders. CE-only throughout. Same 529,210 parameters, learning rate .001 and 110,000 presentations. No per-arm tuning.','','| Training gate | Current gate | Input-prototype routing | Oracle saved route | Any saved route correct |','|---|---:|---:|---:|---:|']
    for arm in args.arms:
        rows=[r for r in records if r['arm']==arm]
        vals=[np.mean([100*r['scores'][k]['accuracy'] for r in rows]) for k in ['current','inferred','oracle']]
        anyval=np.mean([100*r['scores']['bank_any_correct'] for r in rows])
        lines.append(f'| {arm} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | {anyval:.2f}% |')
    lines += ['', 'Oracle uses true class pair only to select a saved route. All ten logits compete. No model weights are restored. Any-route-correct uses ground truth to choose among the five archived routes and is diagnostic only.', '', 'Activation WTA uses one hard winner per neuron based on mean absolute activation over the current minibatch. It recomputes winners before every training update, freezes the last gate for current-gate inference, and archives the last gate at each task boundary. It uses no error feedback or task identity to select training winners. There is no load-balancing penalty. This batch-level WTA differs from per-example activation WTA. Fixed-random uses one route throughout. Random-per-batch resamples one per-neuron configuration each minibatch and archives the last route per task. Task-assigned uses five fixed seeded random configurations, which overlap because each neuron has only four branches. Training task identity selects these configurations. None of these controls protects the shared classifier or optimizer state.', '', 'The unused error buffer remains allocated in non-trace controls for architectural consistency, but is never read by their gates or updated. Identical initial weights and minibatch orders are checked. Fixed-route identity, task-assigned archives, random coverage, disjoint split indices, presentation counts and immutable inference are asserted.', '', 'A low oracle result with random routing only concerns the stored configurations. It does not prove no other configuration can recover a task. This single-seed experiment cannot establish statistical significance.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
