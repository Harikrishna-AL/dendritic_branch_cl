import json, statistics as S, sys
from scipy import stats
bench = sys.argv[1]; runs = json.load(open(f'results/metaplastic_control_20261107/{bench}.json'))
seeds = sorted({r['seed'] for r in runs}); plain = 'mlp' if bench == 'mnist' else 'cnn'
def sm(rs, f): return [S.mean(f(r) for r in rs if r['seed'] == s) * 100 for s in seeds]
def arm(a, al): return [r for r in runs if r['arm'] == a and r['alpha'] == al and r['seed'] in seeds]
alphas = sorted({r['alpha'] for r in runs if r['arm'] == plain and r['alpha']})
print('alpha  valCIL  valTIL  loss  frozen  (MLP/CNN meta, val)')
for al in alphas:
    rs = arm(plain, al)
    if len(rs) < 6: continue
    print(al, round(S.mean(sm(rs, lambda r: r['val']['plain']['cil'])), 2), round(S.mean(sm(rs, lambda r: r['val']['plain']['til'])), 2), round(S.mean(sm(rs, lambda r: r['til_retention_loss_val'])), 1), round(S.mean(r['frozen_frac_end'] for r in rs), 3))
full = [al for al in alphas if len(arm(plain, al)) == 6]
best_cil = max(full, key=lambda al: S.mean(sm(arm(plain, al), lambda r: r['val']['plain']['cil'])))
best_til = max(full, key=lambda al: S.mean(sm(arm(plain, al), lambda r: r['val']['plain']['til'])))
rows = [('branch (image-only)', arm('branch', 0), 'soft'), ('branch (true route)', arm('branch', 0), 'oracle'), (plain + ' plain', arm(plain, 0), 'plain'),
        (f'{plain} meta a={best_cil} (val CIL)', arm(plain, best_cil), 'plain'), (f'{plain} meta a={best_til} (val TIL)', arm(plain, best_til), 'plain')]
print('\nTEST  CIL  |  TIL  |  retention loss (val TIL pts)')
for name, rs, k in rows:
    c = sm(rs, lambda r: r['test'][k]['cil']); t = sm(rs, lambda r: r['test'][k]['til']); l = sm(rs, lambda r: r['til_retention_loss_val'])
    print(f'{name:28s} {S.mean(c):6.2f} ± {S.stdev(c):4.2f} | {S.mean(t):6.2f} ± {S.stdev(t):4.2f} | {S.mean(l):5.1f}')
def ci(a, b):
    d = [x - y for x, y in zip(a, b)]; m = S.mean(d); h = stats.t.ppf(.975, 2) * S.stdev(d) / 3 ** .5; return f'{m:+.2f} [{m - h:+.2f}, {m + h:+.2f}]'
B = arm('branch', 0)
for al, lab in [(best_cil, 'valCIL'), (best_til, 'valTIL')]:
    M = arm(plain, al)
    print(f'branch soft - meta({lab}) CIL', ci(sm(B, lambda r: r['test']['soft']['cil']), sm(M, lambda r: r['test']['plain']['cil'])),
          '| TIL', ci(sm(B, lambda r: r['test']['soft']['til']), sm(M, lambda r: r['test']['plain']['til'])),
          '| branch oracle TIL - meta TIL', ci(sm(B, lambda r: r['test']['oracle']['til']), sm(M, lambda r: r['test']['plain']['til'])))
