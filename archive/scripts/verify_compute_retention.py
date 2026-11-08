"""Verify retention diagnostics on the parameter/compute comparison checkpoints."""
import json
import statistics
from pathlib import Path

import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.split_mnist_usefulness import make
from bdh_cl.scripts.split_mnist_retention_diagnostic import (
    features, head_logits, pair_mask_score, probe, score,
)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    out = Path('results/split_mnist_compute_comparison_20261013')
    runs = json.loads((out / 'results.json').read_text())
    prior = json.loads(Path('results/split_mnist_retention_diagnostic_20261011/results.json').read_text())
    raw = load_raw('data_cache/mnist')
    x = torch.tensor(raw['train_images'].copy(), dtype=torch.float32).flatten(1) / 255
    tx = torch.tensor(raw['test_images'].copy(), dtype=torch.float32).flatten(1) / 255
    y = torch.tensor(raw['train_labels'].copy(), dtype=torch.long)
    ty = torch.tensor(raw['test_labels'].copy(), dtype=torch.long)
    records = []
    for run in runs:
        seed, arm = run['seed'], run['arm']
        torch.manual_seed(seed)
        random = make(arm, 'hard')
        model = make(arm, 'hard')
        model.load_state_dict(torch.load(out / f'{seed}_{arm}.pt', weights_only=True))
        before = {k: v.clone() for k,v in model.state_dict().items()}
        row = {'seed': seed, 'arm': arm,
               'original': score(head_logits(model, features(model, tx)), ty),
               'pair_mask_oracle': pair_mask_score(head_logits(model, features(model, tx)), ty),
               'final_features_fresh_readout': probe(model, x, y, tx, ty),
               'random_features_fresh_readout': probe(random, x, y, tx, ty)}
        if arm == 'local':
            routes = torch.tensor(run['routes'])
            z = features(model, tx, 'saved_routes', routes, ty // 2)
            row['saved_routes_original_readout_oracle'] = score(head_logits(model,z), ty)
            row['all_branches_fresh_readout'] = probe(model,x,y,tx,ty,'all_branches')
            row['random_all_branches_fresh_readout'] = probe(random,x,y,tx,ty,'all_branches')
        reference = next(r for r in prior if r['seed']==seed and r['arm']==arm)
        for key, value in row.items():
            if isinstance(value, dict):
                assert abs(value['all_accuracy']-reference[key]['all_accuracy']) < 1e-7, (seed,arm,key)
        assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
        records.append(row)
        print(seed,arm,'all diagnostics reproduced',flush=True)
    (out/'retention_diagnostics.json').write_text(json.dumps(records,indent=2))
    (out/'retention_diagnostic_source.py').write_text(Path(__file__).read_text())
    rows = [('original','Ordinary final classifier'),
            ('saved_routes_original_readout_oracle','Saved configurations + oracle pair selection; original classifier'),
            ('pair_mask_oracle','Oracle restricts original outputs to correct pair'),
            ('final_features_fresh_readout','Frozen final features + fresh classifier'),
            ('random_features_fresh_readout','Random features + fresh classifier'),
            ('all_branches_fresh_readout','All 96 final branch features + fresh classifier'),
            ('random_all_branches_fresh_readout','All 96 random branch features + fresh classifier')]
    lines=['# Retained-information diagnostics', '',
           'Verified again on the compute-comparison checkpoints: all six runs reproduce the prior diagnostics. '
           'All model weights and memory buffers stay frozen. Values are three-seed means on the official test set.', '',
           '| Diagnostic | MLP: all / old digits | Branch: all / old digits |', '|---|---:|---:|']
    for key,label in rows:
        cells=[]
        for arm in ['mlp','local']:
            rs=[r[key] for r in records if r['arm']==arm and key in r]
            cells.append('Not applicable' if not rs else ' / '.join(
                f'{100*statistics.mean(r[k] for r in rs):.2f}%' for k in ['all_accuracy','old_accuracy']))
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', 'Old digits are 0–7. Saved configurations are routing indices, not earlier network weights; '
              'the ordinary MLP has no routing configuration to restore. Oracle pair masking gives both models '
              'the true digit pair but only tests discrimination within that pair. Saved-route recovery also '
              'uses that identity to choose an external historical route, retaining ten competing outputs.', '',
              'Fresh classifiers are linear ridge probes (fixed regularization 1), trained using all 60,000 '
              'historical training labels. Feature standardization uses training data only; evaluation uses '
              '10,000 held-out test images. No test tuning. MLP has 95 features, current branch output 24, '
              'all-branch probe 96. Probes change readout capacity and require old labelled data; these are '
              'diagnostics, not autonomous continual-learning solutions.', '',
              'The MLP retains useful digit-discriminating information and within-pair decisions. However, '
              'its final-feature probe does not outperform its random-feature control, so that probe alone '
              'does not demonstrate preservation of learned representations. The branch model exposes earlier '
              'capabilities through saved configurations, but these results do not establish that only branches '
              'retain information. Do not directly interpret 98.02% oracle recovery versus 82.15% refitted-head '
              'accuracy as a matched-protocol architecture advantage.', '']
    report='\n'.join(lines)
    (out/'RETENTION.md').write_text(report)
    main_report=out/'REPORT.md'
    marker='\n## Retention follow-up\n'
    text=main_report.read_text().split(marker)[0]
    main_report.write_text(text+marker+'\nSee [retained-information diagnostics](RETENTION.md) for saved-configuration accuracy, MLP readout recovery, and random-feature controls.\n')
    print(report)


if __name__=='__main__':
    main()
