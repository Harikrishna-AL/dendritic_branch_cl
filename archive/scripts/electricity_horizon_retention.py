"""No-task-ID continual learning across real electricity forecast horizons."""
import argparse
import collections
import copy
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.scripts.electricity_memory_pilot import Forecast, features
from bdh_cl.scripts.electricity_usefulness_pilot import Useful, cohort
from bdh_cl.scripts.electricity_cohort_retention import SharedQuality


ARMS = ('quality_all', 'shared_quality', 'uniform_all', 'predict_winner',
        'shared', 'mlp', 'gru')


def build(arm):
    if arm == 'shared_quality':
        return SharedQuality()
    if arm in ('mlp', 'gru', 'shared'):
        return Forecast(arm)
    return Useful(arm)


def train(model, opt, d, start, end, horizon):
    y = d['y']
    queue, batch = collections.deque(), []
    for t in range(start, end):
        while queue and queue[0][0] <= t:
            when, ticket, base = queue.popleft()
            if getattr(model, 'feedback', False):
                model.observe(ticket, y[when] - base)
            batch.append((ticket, y[when] - base))
        if len(batch) >= 8:
            opt.zero_grad(set_to_none=True)
            losses = [model.training_loss(ticket, target) if hasattr(model, 'training_loss')
                      else F.mse_loss(model.ticket_prediction(ticket), target)
                      for ticket, target in batch]
            torch.stack(losses).mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            batch = []
        with torch.no_grad():
            _, ticket = model.forecast(features(y, t))
        if t + horizon < len(y):
            queue.append((t + horizon, ticket, y[t].clone()))


def evaluate(model, d, start, end, horizon, carry=False):
    model = copy.deepcopy(model)
    if not carry:
        model.reset()
    y = d['y']
    queue, errors = collections.deque(), []
    feedback = getattr(model, 'feedback', False)
    weights = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for t in range(start, end - horizon):
            while queue and queue[0][0] <= t:
                when, ticket, base = queue.popleft()
                model.observe(ticket, y[when] - base)
            residual, ticket = model.forecast(features(y, t))
            pred = y[t] + residual
            errors.append((pred - y[t + horizon]).square().mean().item())
            if feedback:
                queue.append((t + horizon, ticket, y[t].clone()))
    assert all(torch.equal(a, b) for a, b in zip(weights, model.parameters()))
    return {'mse': statistics.mean(errors), 'first24_mse': statistics.mean(errors[:24]),
            'first_week_mse': statistics.mean(errors[:168]),
            'after_week_mse': statistics.mean(errors[168:]), 'hours': len(errors)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', default='21,22,23')
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--old-horizon', type=int, default=24)
    p.add_argument('--new-horizon', type=int, default=168)
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    d = cohort(0)
    years, months = d['year'], d['month']
    a_end = int(np.flatnonzero((years == 2012) & (months == 10))[0])
    a_test_end = int(np.flatnonzero(years == 2013)[0])
    b_start = a_test_end + 720
    b_end = int(np.flatnonzero((years == 2013) & (months == 10))[0])
    b_test_end = int(np.flatnonzero(years == 2014)[0])
    (out / 'source.py').write_text(Path(__file__).read_text())
    protocol = {
        **vars(args), 'arms': args.arms.split(','), 'clients': d['clients'],
        'task_A': f"Learn {args.old_horizon}-hour-ahead forecasts on 2012 Feb-Sep; held-out October-December 2012.",
        'task_B': f"Continue learning {args.new_horizon}-hour-ahead forecasts on 2013 Feb-Sep; held-out October-December 2013.",
        'no_task_id': 'The model receives the same 25 observed-load features for both horizons. No horizon, task ID, calendar feature, or boundary signal is provided. The task identity is used only by the training target schedule and analysis.',
        'causality': 'Each forecast is trained only when its target is observed at the specified horizon. Branch usefulness feedback is delayed to that same time. Adam .001, update every eight mature tickets, clipping 5; optimizer and runtime state persist at the task switch.',
        'retention': 'Evaluate the old 24-hour horizon after A and after B with frozen slow weights, including cold runtime reset and carried B runtime context. Evaluate the new 168-hour horizon after B. Slow weights are verified unchanged during audits.',
        'primary_ablation': 'quality_all stores per-customer, per-output-branch factual error; shared_quality pools error over all 16 customer streams and stores one shared layer-level branch vector. uniform_all removes branch selection. All have identical slow architecture and all-head training objective.',
        'limitations': 'This is a real-data, task-label-free multi-horizon continual-learning stress test, not proof of spontaneous task-boundary detection. Three seeds are a pilot. One cohort and fixed horizon order limit generality.'
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    records = []
    for seed in map(int, args.seeds.split(',')):
        for arm in args.arms.split(','):
            torch.manual_seed(seed)
            model = build(arm)
            model.reset()
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            train(model, opt, d, 720, a_end, args.old_horizon)
            state, optstate = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
            recent = copy.deepcopy(model.recent)
            row = {'seed': seed, 'arm': arm,
                   'parameters': sum(x.numel() for x in model.parameters()),
                   'runtime_buffer_bytes': sum(x.numel() * x.element_size() for x in model.buffers()),
                   'after_A': evaluate(model, d, a_end, a_test_end, args.old_horizon)}
            model.load_state_dict(state)
            model.recent = copy.deepcopy(recent)
            opt.load_state_dict(optstate)
            train(model, opt, d, b_start, b_end, args.new_horizon)
            row['after_B'] = {'old_cold': evaluate(model, d, a_end, a_test_end, args.old_horizon),
                              'old_carry': evaluate(model, d, a_end, a_test_end, args.old_horizon, carry=True),
                              'new': evaluate(model, d, b_end, b_test_end, args.new_horizon)}
            row['forgetting_cold'] = row['after_B']['old_cold']['mse'] - row['after_A']['mse']
            row['forgetting_return'] = row['after_B']['old_carry']['mse'] - row['after_A']['mse']
            records.append(row)
            (out / 'results.json').write_text(json.dumps(records, indent=2))
            print(seed, arm, 'old-horizon change', round(row['forgetting_cold'], 4),
                  'new-horizon MSE', round(row['after_B']['new']['mse'], 4), flush=True)

    lines = [f'# Electricity {args.old_horizon}h-to-{args.new_horizon}h continual-learning ablation', '',
             f"{len(args.seeds.split(','))} fresh seeds. Old task is {args.old_horizon}-hour-ahead forecasting; new task is {args.new_horizon}-hour-ahead forecasting. Positive change means old-horizon forgetting.", '',
             '| Model | Params | Runtime buffer bytes | Old before | Old after, cold | Change | Old after, carry | New 168h after |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in args.arms.split(','):
        rs = [r for r in records if r['arm'] == arm]
        vals = [statistics.mean(r['after_A']['mse'] for r in rs),
                statistics.mean(r['after_B']['old_cold']['mse'] for r in rs),
                statistics.mean(r['forgetting_cold'] for r in rs),
                statistics.mean(r['after_B']['old_carry']['mse'] for r in rs),
                statistics.mean(r['after_B']['new']['mse'] for r in rs)]
        lines.append('| ' + arm + ' | ' + str(rs[0]['parameters']) + ' | ' +
                     str(rs[0]['runtime_buffer_bytes']) + ' | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
    lines += ['', 'Primary ablation: quality_all versus shared_quality isolates per-stream branch utility state against one pooled layer-level utility vector, with the same weights and training objective. Negative change is backward transfer. The prediction horizons are used only as objective/evaluation labels, not network inputs.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
