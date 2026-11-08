"""Seed-level statistics: runs are averaged over task orders within a seed first."""
from __future__ import annotations

import statistics

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}


def seed_means(runs: list[dict], value) -> dict[int, float]:
    """value(run) -> float; returns {seed: mean over that seed's runs (orders)}."""
    seeds = sorted({r['seed'] for r in runs})
    return {s: statistics.mean(value(r) for r in runs if r['seed'] == s) for s in seeds}


def mean_sd(values) -> tuple[float, float]:
    v = list(values); return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.)


def paired_ci(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, float]:
    """Mean of a - b over shared seeds with a 95% t-interval."""
    d = [a[s] - b[s] for s in sorted(set(a) & set(b))]; m = statistics.mean(d)
    h = T95.get(len(d), 1.96) * statistics.stdev(d) / len(d) ** .5 if len(d) > 1 else float('nan')
    return m, m - h, m + h


def fmt(m, sd=None, scale=100, digits=2):
    return f'{scale * m:.{digits}f}' + (f' ± {scale * sd:.{digits}f}' if sd is not None else '')
