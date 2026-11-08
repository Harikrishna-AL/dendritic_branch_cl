"""Task-label-free sequential transfer across disjoint Electricity customer cohorts.

Tests whether branch-local predictive-usefulness memory reduces forgetting when
slow weights are trained on one set of electricity customers and then another.
The only difference between quality_all and shared_quality is whether the
delayed per-branch error EMA is stored per customer stream or pooled across the
layer. All model weights, data, order, and optimizer settings are matched.
"""
import argparse
import copy
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from bdh_cl.scripts.electricity_memory_pilot import Forecast
from bdh_cl.scripts.electricity_retention_pilot import audit
from bdh_cl.scripts.electricity_usefulness_pilot import Useful, cohort, run


class SharedQuality(Useful):
    """One common branch-utility vector shared across all customer streams."""

    def __init__(self, streams=16):
        super().__init__('quality_all', streams)
        self.error = self.error[:1].clone()

    @torch.no_grad()
    def observe(self, ticket, target):
        # Pool factual errors across customers, retaining branch-specific scores.
        pooled = (ticket[3] - target[:, None]).square().mean(0, keepdim=True)
        self.error.mul_(.95).add_(pooled, alpha=.05)


ARMS = ('quality_all', 'shared_quality', 'uniform_all', 'predict_winner',
        'shared', 'mlp', 'gru')


def build(arm):
    if arm == 'shared_quality':
        return SharedQuality()
    if arm in ('mlp', 'gru', 'shared'):
        return Forecast(arm)
    return Useful(arm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', default='13,14,15')
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--a-offset', type=int, default=0)
    p.add_argument('--b-offset', type=int, default=32)
    args = p.parse_args()

    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    a, b = cohort(args.a_offset), cohort(args.b_offset)
    assert not set(a['clients']) & set(b['clients']), 'cohorts must be disjoint'
    assert len(a['y']) == len(b['y'])

    def windows(d):
        years, months = d['year'], d['month']
        a_end = int(np.flatnonzero((years == 2012) & (months == 10))[0])
        a_test_end = int(np.flatnonzero(years == 2013)[0])
        b_start = a_test_end + 720
        b_end = int(np.flatnonzero((years == 2013) & (months == 10))[0])
        b_test_end = int(np.flatnonzero(years == 2014)[0])
        return a_end, a_test_end, b_start, b_end, b_test_end

    a_end, a_test_end, _, _, _ = windows(a)
    _, _, b_start, b_end, b_test_end = windows(b)
    (out / 'source.py').write_text(Path(__file__).read_text())
    protocol = {
        **vars(args), 'arms': args.arms.split(','),
        'cohort_A': a['clients'], 'cohort_B': b['clients'],
        'sequence': 'Train cohort A on 2012 Feb-Sep; then train disjoint cohort B on 2013 Feb-Sep with the same model, optimizer, and runtime state. No cohort/task label, boundary signal, or customer ID is fed to the network.',
        'calibration': 'Each customer series is normalized using its own first 720 hours in 2012. Forecast inputs are 24 observed hours plus the matching hour one week earlier; target is 24 hours ahead.',
        'retention': 'Score cohort A October-December 2012 before and after cohort B training, and cohort B October-December 2013 after its training. Audit slow weights are frozen; quality arms receive factual forecast errors after 24 hours. Both cold-reset and carried-state A-return audits are saved.',
        'primary_ablation': 'quality_all stores a separate 32-branch error EMA for every stream; shared_quality pools the same delayed factual errors over streams and stores a single 32-branch layer vector. Both have identical slow weights, training loss, optimizer, and per-branch output; only the error-memory scope changes. uniform_all is the nonselective branch control.',
        'other_controls': 'predict_winner uses activation predictability instead of factual branch usefulness; shared is the same-parameter branch model with full shared input/hidden moment state; MLP and GRU are dense baselines.',
        'learning': 'Adam .001, update every eight matured 24-hour targets, gradient clipping 5; one chronological pass; model and optimizer state persist across A-to-B; pending cross-window targets are dropped at segment boundaries.',
        'interpretation_limit': 'This is a real-data customer-domain transfer stress test, not proof of natural task discovery. The data are genuine electricity loads, but the cohort boundary is used only by the experimenter to schedule training and scoring. The disjoint cohorts use distinct per-customer calibration normalization. Results may reflect customer distribution shift as well as continual-learning interference.'
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    records = []
    for seed in map(int, args.seeds.split(',')):
        for arm in args.arms.split(','):
            torch.manual_seed(seed)
            model = build(arm)
            model.reset()
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            run(model, opt, a, 720, a_end, True, False)
            state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(opt.state_dict())
            recent = copy.deepcopy(model.recent)
            row = {'seed': seed, 'arm': arm,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'buffer_bytes': sum(x.numel() * x.element_size() for x in model.buffers()),
                   'after_A': {'old': audit(model, a, a_end, a_test_end),
                               'new': audit(model, b, b_end, b_test_end)}}

            model.load_state_dict(state)
            model.recent = copy.deepcopy(recent)
            opt.load_state_dict(opt_state)
            run(model, opt, b, b_start, b_end, True, False)
            row['after_B'] = {'old_cold': audit(model, a, a_end, a_test_end),
                              'old_carry': audit(model, a, a_end, a_test_end, carry=True),
                              'new': audit(model, b, b_end, b_test_end)}
            row['forgetting_cold'] = row['after_B']['old_cold']['mse'] - row['after_A']['old']['mse']
            row['forgetting_return'] = row['after_B']['old_carry']['mse'] - row['after_A']['old']['mse']
            row['new_gain'] = row['after_A']['new']['mse'] - row['after_B']['new']['mse']
            records.append(row)
            (out / 'results.json').write_text(json.dumps(records, indent=2))
            print(seed, arm, 'forgetting', round(row['forgetting_cold'], 4),
                  'new gain', round(row['new_gain'], 4), flush=True)

    lines = [
        '# Electricity disjoint-cohort retention ablation', '',
        'Three fresh initialization seeds. Positive forgetting means cohort-A held-out MSE worsened after cohort-B training. Audits freeze slow weights; causal usefulness feedback may adapt during audit.', '',
        '| Model | Params | Error-state bytes | A before | A after, cold | Forgetting, cold | A after, carry | Forgetting, carry | B after | B gain |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for arm in args.arms.split(','):
        rs = [r for r in records if r['arm'] == arm]
        def avg(path):
            return statistics.mean(r[path[0]][path[1]]['mse'] for r in rs)
        vals = [avg(('after_A', 'old')), avg(('after_B', 'old_cold')),
                statistics.mean(r['forgetting_cold'] for r in rs),
                avg(('after_B', 'old_carry')),
                statistics.mean(r['forgetting_return'] for r in rs),
                avg(('after_B', 'new')),
                statistics.mean(r['new_gain'] for r in rs)]
        lines.append('| ' + arm + ' | ' + str(rs[0]['parameters']) + ' | ' +
                     str(rs[0]['buffer_bytes']) + ' | ' + ' | '.join(f'{x:.4f}' for x in vals) + ' |')
    lines += ['', 'The primary comparison is quality_all vs shared_quality: same forecasting network and objective, with per-stream branch utility memory versus one pooled layer utility vector. Negative forgetting is backward transfer. The shared full-moment arm uses more context state than branch-local memory; it is a broad control, not an exact state-budget match.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
