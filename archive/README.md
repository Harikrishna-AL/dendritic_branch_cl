# Archive

These are the earlier experiments (October–November 2026) that led to the current code. They are kept as-is for the record, including the negative results. **Nothing here is needed to reproduce the top-level README.**

The project started as a variant of the BDH model (hence the `bdh_cl` name), and it later moved to the dendritic-branch-routing approach in `branch_routing/`.

- `v1_code/bdh_cl/`: the previous one-script-per-experiment code behind the November reports. `branch_routing/` reimplements its main models and training rule. On Split MNIST the reimplementation reproduces the old runs bit for bit. On CIFAR it matches the old code when both run with the same thread count; CPU convolution results depend on the number of threads.
- `v1_results/`: reports and JSON from those scripts. These include the diagnoses that are not rerun in the new code: why an ordinary CNN plus branches fails, the CIFAR access probes, frozen-copy keys, and the earlier MNIST "hybrid5" model and its stream results.
- `scripts/`, `bdh_cl_legacy/`: older experiments and the original BDH-based package. They need path fixes before they run.
- `notes/`: the lab notebook (`NOTES.md`), the dendritic equations and experiment write-ups.
- `reports/`: earlier compiled reports and figures.
- `slurm/`, `run_sweep.sh`: cluster job scripts.
- `MOVED.txt`: every file moved during the reorganisation (old path → new path).

These are kept locally and not committed: `results/` (raw runs), `external/` (third-party clones such as HAT, Mammoth and DCNet), `upstream_bdh/`, `data/`, `tmp/` and `output/`.
