# NOTES — deviations, failures and judgement calls

Running log for the BDH continual-learning harness. Everything that did not go
to plan, or where I had to choose, is recorded here. Phase 1: baseline BDH only,
no dendrites.

Environment: Apple M4, 10 cores, **16 GiB unified memory**, macOS 27.0,
torch 2.8.0, Python 3.13.7, device **MPS** (no CUDA). Full detail in
`bdh_cl/results/env.json`. Runs use the venv at
`/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv`.

Upstream pinned at `pathwaycom/bdh` commit `2b0d7a45b058d4309c84a10e0768d541fe18bdc2`,
cloned to `bdh-d/upstream_bdh/` and **never edited**.

---

## Step 0 — reconnaissance

### What `train.py` does

| aspect | what it actually does |
|---|---|
| data | tiny Shakespeare (1.1 MB), downloaded on first run to `input.txt` |
| tokenizer | **none** — the file is `np.memmap`'d as `uint8` and bytes are the tokens, vocab 256 |
| split | first 90% train, last 10% val, by position |
| batching | `BATCH_SIZE=32` random windows of `BLOCK_SIZE=512`; `y` is `x` shifted by 1 |
| optimizer | `AdamW(lr=1e-3, weight_decay=0.1)`, default betas, **no LR schedule**, **no grad clipping** |
| steps | `MAX_ITERS=3000` |
| loss | `F.cross_entropy` over all positions |
| logging | mean loss printed every 100 steps to stdout; no eval, no checkpointing, no val loss (the `eval()` function is a stub that only calls `model.eval()`) |
| precision | picks `bfloat16` on CUDA, else `float16`; autocast only on CUDA; `GradScaler` enabled whenever dtype is fp16 |
| compile | `torch.compile(model)` unconditionally |
| finish | generates 100 bytes from the prompt `"To be or "` with `top_k=3` |

Device selection is `cuda if available else cpu` — **MPS is never selected**,
only mentioned in a comment.

### Parameter count

The three weight matrices are allocated **once** and reused at every level of
the layer loop (`bdh.py:122-144` reads `self.encoder`, `self.encoder_v`,
`self.decoder` inside `for level in range(C.n_layer)`). So:

```
encoder    nh * D * N   = mult * D^2
encoder_v  nh * D * N   = mult * D^2
decoder    (nh*N) * D   = mult * D^2
embed      vocab * D
lm_head    D * vocab
LayerNorm  0             (elementwise_affine=False)
---------------------------------------------------
total = 3 * mult * D^2 + 2 * vocab * D = 3*n*d + 2*vocab*d,  n = nh*N = mult*d
```

At the default config (`D=256, mult=128, nh=4`): `N = 8192`, `n = 32768`,
**25,296,896 parameters ≈ 25.3 M** — matching the ≈3·n·d ≈ 25 M expectation in
the brief.

> **`n_layer` does not change BDH's parameter count at all** — only its depth of
> computation. This is a genuine architectural difference from a Transformer and
> it matters for parameter matching: BDH gets depth for free, so the matched GPT
> baseline ends up much deeper or much wider. It is also worth flagging for
> phase 2 — a dendritic BDH inherits the same weight sharing, so `k` branches
> cost `k×` on the encoders *once*, not once per layer.
> Verified analytically and empirically in `tests/test_bdh_equivalence.py::test_analytic_param_count_matches_module`.

### The attention implementation

`bdh.py:56-74`. Per layer:

```python
scores = (QR @ KR.mT).tril(diagonal=-1)   # B, nh, T, T
return scores @ V
```

with `Q = K = x_sparse` (shape `B, nh, T, N`, ReLU output so **non-negative**)
and `V = x` (shape `B, 1, T, D`, broadcast over heads).

Four things worth stating explicitly:

1. **No softmax and no `1/sqrt(d)` scaling.** Scores are raw inner products of
   two non-negative vectors, so every weight is ≥ 0 but unnormalised — the row
   sums grow with `T`. The `self.ln(yKV)` on the next line is what keeps this
   from exploding.
2. **`tril(diagonal=-1)` excludes the diagonal**, so a token attends only to
   *strictly* earlier tokens and never to itself. Position 0 therefore gets an
   exactly-zero attention output. Our GPT baseline uses the standard
   `is_causal=True` (diagonal included); I kept each model's native convention
   rather than crippling either, and note the asymmetry here.
3. **Full `T×T` materialisation.** No flash/chunked path, so memory is
   `O(B · nh · T²)` and, much worse, the `(B, nh, T, N)` sparse tensors are the
   real cost.
4. RoPE uses `theta=2**16` and a `quantize(t, q=2)` step so adjacent frequency
   pairs share a phase, plus a `1/(2π)` factor — different from standard
   `theta=10000` RoPE, but it is upstream's choice and we do not touch it.

**Memory cost at the context lengths we use** (fp32, activations saved for
backward: `x_sparse`, `y_sparse`, `xy_sparse` per layer, plus scores):

| B | T | activation memory |
|---|---|---|
| 32 | 512 (**upstream default**) | **36.8 GiB** |
| 16 | 256 | 9.1 GiB |
| 8 | 256 | 4.6 GiB |
| 8 | 96 (our synthetic runs) | 0.15 GiB |

The `(B, nh, T, N)` tensors dominate: at the default config `N = 8192`, so one
of them is 2.0 GiB at `B=32, T=512`, and there are three per layer across 6
layers. The `T×T` score matrix is comparatively negligible (0.12 GiB).

### FAILURE: upstream `train.py` cannot run unchanged on this machine

Recorded because the brief asks for the repo's own `train.py` for a few hundred
steps.

* **Attempt 1** — `python train.py`, 900 s timeout. Exited after 4:15 with *no
  output whatsoever*. Cause: stdout was block-buffered into a file and the
  process died before flushing. Diagnosis was wrong-footed by this; the run
  produced nothing usable.
* **Attempt 2** — `python -u train.py` (file untouched, `-u` only defeats
  buffering). Printed `Using device: cpu with dtype float16` and then made no
  progress for ~8 minutes: 0.74 GiB RSS, **2% CPU**, and system swap climbed to
  **37 GB in use**. That is not compute, it is thrashing, and 37 GB matches the
  36.8 GiB activation estimate above almost exactly. Killed manually.

**Conclusion: the upstream default config (25 M params, B=32, T=512, CPU) needs
~37 GiB and does not fit in 16 GiB.** This is a hardware limit, not a bug in
the repo. Log: `bdh_cl/results/logs/upstream_train_unchanged.log`.

### Step 0.2 as actually satisfied

`bdh_cl/scripts/smoke_upstream.py` imports upstream `train.py` **as a module**
(its `main` is `__main__`-guarded, so importing does not train) and uses
upstream's own `BDH_CONFIG`, `get_batch`, `LEARNING_RATE` and `WEIGHT_DECAY`.
Only module-level hyperparameters are rebound from outside. **The file is not
edited.** Deviations, all forced by the hardware:

1. `BATCH_SIZE` 32→16, `BLOCK_SIZE` 512→256 (see memory table).
2. `torch.compile` skipped — slow/unhelpful for this graph on MPS.
3. fp16 `GradScaler` skipped, fp32 on MPS instead. Upstream builds
   `GradScaler(device="cpu", enabled=True)`, a CUDA-oriented path.
4. Gradient clipping at 1.0 added (upstream has none). Applied identically to
   both models everywhere, so it cannot favour either.
5. Device forced to MPS.

**Result (`results/step0_upstream_smoke_tiny.json`), `tiny` preset, 300 steps:**
loss **4.056 → 1.606**, monotone decreasing — *loss decreases, confirmed*.
Throughput **8,485 tokens/sec**.

**`small` (the upstream default config) was attempted and abandoned.** At
B=8/T=256 on MPS it reported 322 tok/s at step 0 and then stalled — no step-25
line after ~20 minutes, 6% CPU, swap at 11.6 GB. Killed. I did **not** get a
loss curve for `small`, and do not report one. Throughput for `small` is
therefore unmeasured beyond that single 322 tok/s figure, which I treat as
unreliable.

### Measured throughput (`results/benchmark_tiny.json`)

Median over 6 steps of a full forward+backward+optimizer step, T=96, MPS:

| model | B=8 | B=16 | B=32 |
|---|---|---|---|
| BDH tiny (1.64 M) | 8,610 tok/s | 8,898 | 9,018 |
| GPT tiny (1.64 M) | 25,812 tok/s | 44,652 | 50,207 |

**At matched parameters BDH is ~5.6× slower per step than the Transformer** on
this hardware (341 ms vs 61 ms at B=32). Expected: BDH's `N=1024` latent width
per head, three elementwise tensors of shape `(B, nh, T, N)` per layer, and an
unfused `T×T` matmul, against a fused SDPA kernel for GPT. This is why a
compute-matched comparison (brief, Step 3) is not just a nice-to-have — at
equal wall-clock GPT gets ~5× the steps.

### Config presets (Step 0.4)

| preset | BDH | params | matched GPT | params | rel |
|---|---|---|---|---|---|
| `tiny` | L=4, d=128, nh=4, mult=32 (N=1024, n=4096) | 1,638,400 | L=8, d=128, nh=4 | 1,640,576 | +0.13% |
| `small` | L=6, d=256, nh=4, mult=128 (N=8192, n=32768) — **upstream default** | 25,296,896 | L=8, d=512, nh=8 | 25,436,672 | +0.55% |
| `base` | L=6, d=512, nh=8, mult=128 (N=8192, n=65536) | 100,925,440 | L=10, d=896, nh=14 | 96,815,488 | −4.07% |

`base` is defined but **almost certainly unusable on this machine** — 100 M
params plus BDH's activation profile. It is there so the presets are complete,
not because I expect to run it. My first `base` GPT guess (L=12, d=768) was
−15.4% off and was replaced using the `match_gpt_to_bdh` search.

Matching leaves the *shape* free: BDH's shared weights mean `small` can be
matched by either `L=32, d=256` (equal width, +0.07%) or `L=8, d=512` (+0.55%).
I chose the latter as the default because a 32-layer byte-level Transformer at
d=256 is an unusual and slow shape; `match_gpt_to_bdh(..., prefer="width")`
gives the equal-width alternative and it is a one-line change. **Judgement
call, and it does affect the comparison** — depth and width are not neutral for
forgetting.

---

## Judgement calls in the harness

**GPT baseline design.** Byte vocab 256 (same as BDH), **RoPE instead of learned
position embeddings**, no weight tying. RoPE because learned `wpe` would make
the parameter count depend on `block_size`, which breaks parameter matching the
moment Step 4 evaluates beyond the training length. Consequence: the GPT
baseline is not literally nanoGPT.

**Optimizer.** AdamW with weight decay on ≥2-D parameters only (norms/biases
excluded), betas (0.9, 0.95), linear warmup then cosine decay to 10% of peak,
restarted per task. Upstream decays everything and has no schedule. Applied
identically to both models.

**Optimizer state between tasks.** Reset by default (`TrainSpec.
reset_optimizer_between_tasks=True`), per the brief. The "keep state" variant
is a config flag; the comparison is pending.

**Wrapper vs upstream.** `BDHWrapper.forward` re-walks upstream's layer loop
rather than calling it, because the ReLUs are functional calls, not submodules,
so no forward hook can see `x_sparse`/`y_sparse`. The brief allows a copy only
with proof, so `tests/test_bdh_equivalence.py` asserts **exact** equality
(`atol=0, rtol=0`) of logits and loss against `upstream_bdh.BDH` in both eval
and train mode, and near-exact equality of **gradients** for every parameter.
The upstream module is kept as `self.core` and remains the sole parameter owner,
so checkpoints stay upstream-loadable. 10 tests, all passing.

**Loss masking.** `masked_cross_entropy(logits, targets, mask=None)` is
bit-identical to upstream's `F.cross_entropy`; a test pins that. Step 2B needs
the mask, upstream does not have one.

**Synthetic task format.** `<tag>|<input>><output>\n`, loss on the output bytes
**including the terminating newline**. The brief says "only output bytes"; I
include the `\n` because without it free-running generation has no stopping
signal and exact-match scoring becomes ill-defined. Flagging it as a deliberate
reading.

**Exact match is measured by free-running greedy generation**, not teacher
forcing — teacher forcing would credit a model for continuing a prefix it could
never have produced. Because neither model takes a padding mask, left-padding
would corrupt the context, so examples are **bucketed by exact prompt length**
and each bucket is generated separately. A cheap teacher-forced masked accuracy
is reported alongside, and is used for the dense in-training curves (a full
evaluation every `eval_every` steps costs more than the training it measures).

**Held-out sets are not provably disjoint from training.** They are drawn from
the same generator with an evaluation-only seed. For `copy`/`reverse`/`lookup`
the spaces are large (26^4 at the shortest) and collisions are negligible; for
`add` with 2-digit operands and short `sort` the space is small enough that some
overlap is likely. I do not claim strict disjointness.

**Replay buffer: "5%" appears in two senses** — the buffer holds 5% of each
finished task's training examples, *and* 5% of each batch is drawn from it. The
brief says "small replay buffer (5% of past data)", which specifies the first.
Both are 5% here; stated so the number is not over-read.

**Probes for GPT.** BDH's latent neurons are non-negative and sparse by
construction. The closest GPT analogue is the post-GELU MLP hidden state, which
is **dense and signed**. Both are thresholded at 0 to define "active". GPT's
density will come out near 50% — that is the architectural difference under
test, not a broken probe. Overlap is reported both as top-k Jaccard (with k as
a *fraction* of neuron count, so the two widths are comparable) and as cosine
similarity of full mean-activation profiles, which does not depend on k.

**Files added beyond the Step 1 layout:** `config.py` (presets + YAML loading),
`models/param_accounting.py` (Step 7's accounting utility), `cl/tasks.py` (a
uniform Task interface so one runner drives every data family), `cl/evaluate.py`
(the metrics themselves), `scripts/detect_env.py`, `scripts/smoke_upstream.py`,
`scripts/benchmark.py`, `scripts/run_cl.py`.

---

## Step 2A — language streams

Three domains, each truncated to an **equal byte budget of 953,081 bytes**
(= the smallest corpus, German). Since the vocabulary is bytes, equal bytes is
equal tokens. Held-out split is the last 10% by position.

| domain | source | non-ASCII bytes | distinct bytes |
|---|---|---|---|
| `english` | tiny Shakespeare (the corpus upstream trains on) | 0.00% | 65 |
| `python` | the **local** Python 3.13 stdlib `.py` files, sorted, `test` dirs skipped — fully offline | 0.02% | 119 |
| `german` | Project Gutenberg 2229, 2230, 22367, 22555 (Faust I/II, Die Verwandlung, Grimms Märchen), licence header/footer stripped | 3.04% | 109 |

### FAILURE: the optional non-Latin stream could not be sourced

The brief lists a non-Latin script (Malayalam or Hindi) as optional. I tried to
add it and could not, honestly:

* I guessed two Gutenberg ids for Hindi. They downloaded, passed every check I
  had, and produced a corpus of **Esperanto** (`id 55954` is *Stranga heredaĵo*).
  The byte statistics gave it away — 1.02 bytes/char and 3.9% non-ASCII, where
  Devanagari would be ~3 bytes/char and near-100% non-ASCII.
* Gutenberg's plain-text mirror has essentially no Devanagari. Browsing by
  language: `hi` returns nothing, `sa` (Sanskrit) returns one book (id 9000,
  *Sri Vishnu Sahasranaamam*) which is **transliterated into Latin**, 0
  Devanagari characters. `gutendex.com` was unreachable. The five books the
  site's own "hindi" search returns are all English or Tagalog *about* Hindi.

**The stream is dropped, not faked.** Two things came out of this:

1. `language.require_script()` now validates that a corpus is actually in the
   script its name promises (≥30% of characters in the right Unicode block) and
   refuses to load it otherwise. A mislabelled corpus is worse than a missing
   one, because it would have been *reported* as a non-Latin result.
2. `hindi` and `malayalam` load from `data_cache/<name>.txt`, so a corpus can be
   dropped in later with no code change — and the guard still checks it.

Tested in `tests/test_language.py`.

## Finding: overlap is not the mechanism of forgetting (BDH, seed 0)

The single most decision-relevant result so far, recorded here because it
contradicts the hypothesis the project was set up to test.

| strategy | final avg | forgetting | BWT | x_sparse Jaccard | y_sparse Jaccard | density |
|---|---|---|---|---|---|---|
| naive | 0.333 | 0.990 | -0.990 | 0.872 | 0.749 | 39.4% |
| replay 5% | 0.780 | 0.310 | -0.310 | 0.859 | 0.643 | 39.8% |
| joint | 0.990 | 0.010 | +0.015 | 0.868 | 0.701 | 40.7% |

Three things follow.

1. **BDH's sparse code is not sparse, and does not separate tasks.** Density is
   ~40% (not the near-zero the "sparse, positive, high-dimensional" framing
   implies), and the top-k neuron sets of different tasks overlap at Jaccard
   0.87 with full-profile cosine 0.99.

2. **Separation is not required for multi-task competence.** The jointly
   trained model performs all three tasks at 0.99 average *using the same
   neurons for all of them*.

3. **Representational overlap does not cause the forgetting.** Joint and naive
   have statistically indistinguishable overlap (0.868 vs 0.872) and opposite
   continual-learning outcomes (BWT +0.015 vs -0.990). The only difference is
   whether other tasks' gradients are present during training. So the assumed
   causal chain *shared neurons -> interference -> forgetting* does not hold
   here; the damage is sequential gradient overwriting, not representational
   collision.

Consequence for phase 2: a dendritic unit that increases task separation is
attacking something this experiment shows is not the bottleneck. If dendrites
help, the mechanism is more likely to be *protection* (branches gradients do
not reach, gating, usage-based freezing) than separation as such.

Validity checks that make me believe the numbers rather than suspect the
harness:

* All three tasks reach exact-match **1.000** in isolation and random-init is
  **0.000**, so every drop in the matrix is interference, not a task the model
  never learned.
* The 5% replay buffer recovers final average from 0.333 to 0.780, so the
  model has ample capacity for all three tasks -- naive forgetting is not a
  capacity ceiling.
* The overlap probe discriminates: `copy|reverse` (same input alphabet) is
  consistently the highest pair at ~0.93, `copy|sort` (letters vs digits)
  the lowest at ~0.84. It is not returning a constant.
* `y_sparse` (value path) is consistently more task-differentiated than
  `x_sparse` (query path), 0.64-0.75 vs 0.86-0.87 -- a structural asymmetry in
  BDH worth carrying into phase 2.

Caveats: one seed, 1.6M parameters, three tasks, GPT arm not yet in.

## Finding: the Transformer is the better continual learner (seed 0)

Under **naive** finetuning both models saturate at total forgetting
(BDH 0.990, GPT 1.000), so that comparison cannot discriminate between them --
a floor effect, noted at the time as a risk. **Replay is the regime with
headroom**, and there the two separate cleanly:

| seed 0, replay 5% | final avg | forgetting | BWT | copy (final) | reverse (final) |
|---|---|---|---|---|---|
| BDH | 0.780 | +0.310 | -0.310 | 0.740 | 0.600 |
| **GPT** | **0.983** | **0.000** | **0.000** | **1.000** | **1.000** |

With an identical 5% buffer the Transformer forgets **nothing**; BDH loses a
quarter of `copy` and 40% of `reverse`. GPT does this despite a *lower*
single-task ceiling on `sort` (0.94 vs BDH's 1.00), so it is not winning by
being the stronger learner.

This aligns with the probe result rather than contradicting it: GPT's
representations are also more task-separated (Jaccard 0.575 vs 0.872, cosine
0.889 vs 0.993) and marginally sparser (32.0% vs 35.9%/43.0% active). Two
independent measurements, same direction.

**Provisional phase-1 answer:** at this scale baseline BDH is *worse* at
continual learning than a parameter-matched Transformer, and its
representations are *less* task-separated -- despite sparse, positive,
high-dimensional activations being the property meant to deliver both.

Relation to the "overlap is not the mechanism" note above: within BDH, joint
and naive have the same overlap and opposite outcomes, so overlap alone does
not explain forgetting *within* an architecture. Across architectures, lower
overlap and better retention co-occur. With one architecture pair and one seed
that is a correlation, not a cause, and it is written up as such.

**Not yet confirmed: one seed.** Seeds 1-2 decide whether this survives.

## Correction: the "rigid representation" claim was over-read from one seed

While seed 0 was the only data, I wrote that **BDH's overlap is rigid** --
0.859-0.872 across all three strategies -- **while GPT's moves** (0.575 ->
0.246 -> 0.397), and framed that as BDH being unable to reorganise.

Seed 1 does not support the strong form:

| strategy | BDH x_sparse overlap, seed 0 | seed 1 |
|---|---|---|
| naive | 0.872 | 0.895 |
| replay | 0.859 | 0.778 |
| joint | 0.868 | 0.750 |

BDH's across-strategy range is 0.013 on seed 0 but **0.145 on seed 1**. The
representation does shift with training regime; "rigid" was an artifact of a
single draw.

What survives, pending seeds 1-2 of GPT:

* BDH's absolute overlap (0.75-0.90) is consistently far above GPT's
  (0.25-0.58) -- a much larger effect than the seed noise.
* In both BDH seeds the ordering is the same: naive highest, replay/joint
  lower.

**Replay also has large seed variance** and this matters for the headline:
BDH's final average is 0.780 (seed 0) vs 0.903 (seed 1), forgetting 0.310 vs
0.140. GPT seed 0 scored 0.983. The BDH-vs-GPT replay gap may be much smaller
than seed 0 implied, and cannot be stated until GPT seeds 1-2 are in.

By contrast, **naive forgetting replicates almost exactly** (0.990 / 0.995)
and the joint ceiling is stable (0.990 / 0.987).

Lesson, recorded because it happened twice: single-seed structure in this
setup looks more decisive than it is. No architectural claim goes in the
report on fewer than 3 seeds.

## MAJOR CORRECTION: the GPU 3-seed run reverses the replay result

The laptop runs (2 seeds, MPS) led me to report that **GPT retains essentially
everything under replay while BDH loses 14-31%**. The full 3-seed run on CUDA
says the opposite, and more importantly says the comparison was the wrong
shape.

### Per-seed replay, final row (copy / reverse / sort)

| seed | BDH | GPT |
|---|---|---|
| 0 | 0.780 / 0.650 / 1.000 = **0.810** | 1.000 / 0.960 / 0.960 = **0.973** |
| 1 | 0.880 / 0.760 / 1.000 = **0.880** | **0.000** / 0.840 / 0.950 = 0.597 |
| 2 | 0.760 / 0.660 / 1.000 = **0.807** | **0.010** / 0.870 / 0.980 = 0.620 |
| mean | **0.832 ± 0.034** | **0.730 ± 0.172** |

**GPT's replay is bimodal**: it either retains `copy` perfectly or loses it
entirely, and 2 of 3 seeds collapse. **BDH degrades gracefully and
consistently** on every seed. The means favour BDH, but a mean is the wrong
summary for a bimodal variable -- the distributions differ in *kind*.

My earlier claim came from GPT's two lucky seeds. Note that seed 1 gave 0.970
on MPS and 0.597 on CUDA: the same seed, opposite outcomes. This configuration
sits near a bifurcation where floating-point differences flip the result, which
is itself worth reporting.

### What the 3-seed GPU run does support

| run | final | forgetting |
|---|---|---|
| bdh_naive | 0.333 ± 0.000 | 0.980 ± 0.008 |
| bdh_replay | 0.832 ± 0.034 | 0.240 ± 0.050 |
| bdh_joint | 0.993 ± 0.005 | 0.002 ± 0.002 |
| gpt_naive | 0.320 ± 0.012 | 0.988 ± 0.013 |
| gpt_replay | 0.730 ± 0.172 | 0.370 ± 0.248 |
| gpt_joint | 0.943 ± 0.022 | 0.000 ± 0.000 |

* **Naive**: both saturate at total forgetting. No discrimination (floor effect).
* **Joint**: BDH 0.993 > GPT 0.943. BDH is the better multi-task learner.
* **Replay**: BDH higher mean *and* 5x lower variance.

### The overlap finding is robust and points the OTHER way

| run | top-5% Jaccard | cosine | density |
|---|---|---|---|
| bdh naive / replay / joint (x_sparse) | 0.849±.02 / 0.746±.05 / 0.790±.05 | 0.98-0.99 | 38-42% |
| gpt naive / replay / joint | 0.343±.02 / 0.198±.07 / 0.365±.02 | 0.49-0.73 | 30-35% |

**BDH's representations overlap far more than GPT's** (0.61-0.85 vs 0.20-0.37),
a gap many times the seed noise -- yet BDH forgets *less* and more reliably.

**More overlap, less forgetting.** Separation and retention point in opposite
directions across architectures, which kills the assumed chain
*shared neurons -> interference -> forgetting* at the between-architecture
level too, not just within BDH.

### Methodological lesson (third occurrence)

Single- and two-seed structure in this setup has now misled me three times:
the "rigid representation" claim, the replay gap, and the tiny-scale sparsity
comparison. Nothing architectural goes in the report on fewer than 3 seeds,
and platform (MPS vs CUDA) counts as a factor, not a detail.

## PHASE 2 RESULT: dendrites help, but not for the reason I predicted

Full 3-seed run on H100 (`results/cluster/exp4`), 3 tasks, replay strategy.
All dendritic variants use k=4 branches and 1,408 neurons.

| model | params | neurons | replay final | forgetting |
|---|---|---|---|---|
| bdh (baseline) | 1.64M | 4,096 | 0.814 ± 0.049 | +0.257 ± 0.062 |
| bdh_narrow (control) | 0.61M | 1,408 | 0.764 ± 0.035 | +0.327 ± 0.054 |
| **bdh_den_gated** | 1.69M | 1,408 | **0.932 ± 0.016** | **+0.097 ± 0.027** |
| bdh_den_sum | 1.69M | 1,408 | 0.923 ± 0.021 | +0.108 ± 0.038 |
| bdh_den_max | 1.69M | 1,408 | 0.868 ± 0.020 | +0.190 ± 0.032 |

Under **naive** all five sit at 0.332-0.333 (floor effect, no discrimination).

**The control rules out the width confound.** Parameter matching forced the
dendritic models narrower, so the gain might have been "1,408 neurons trains
better". It is not: `bdh_narrow` at the same 1,408 neurons is *worse* than
baseline (0.764 vs 0.814). At equal neurons, adding 4 branches goes
0.764 -> 0.932. At equal parameters, spending them on branches beats spending
them on neurons (0.932 vs 0.814).

### Branches specialise by task -- but only when the regime demands it

`same_branch_frac` = fraction of neurons whose dominant branch is shared by two
tasks, measured against **each model's own random init** (an untrained unit
already scores ~0.86 from weight-norm asymmetry, so 1/k is not a usable
reference).

| variant | naive: init -> final | replay: init -> final |
|---|---|---|
| gated | 0.855 -> 0.887 (**+0.032**) | 0.855 -> 0.773 (**-0.082**) |
| sum | 0.859 -> 0.869 (+0.011) | 0.859 -> 0.765 (**-0.094**) |
| max | 0.875 -> 0.911 (+0.036) | 0.875 -> 0.810 (**-0.065**) |

The sign flips with the training regime, in all 3 variants across 3 seeds:
under replay (model must hold several tasks) tasks move onto *different*
branches; under naive (one task at a time) they converge onto the *same*
branch. **Nothing in the loss asks for this.** Branch entropy is 0.58-0.65
normalised and dead branches only 1.7-3.9%, so branches are genuinely used,
not collapsed.

### My mechanistic prediction was wrong

`DENDRITIC_EQUATIONS.md` argues `max` should win, because winner-take-all sends
gradient to exactly one branch and therefore *shields* an unused branch from
another task's gradients. Measured: **`max` is the worst of the three**
(0.868), and **`sum` -- which has no shielding at all, every active branch
receives gradient -- is within 0.01 of `gated`** (0.923 vs 0.932) and shows the
*largest* branch divergence.

So gradient protection is not the mechanism. The gain is more likely the extra
per-neuron expressivity: `sum_c relu(w_c . x)` is a richer positive function
than a single `relu(w . x)`, which buys more with the same parameters than
widening does. The user's argument for `gated` -- that it can use branch
structure or behave as a plain neuron, whichever the task needs -- is the
variant that in fact wins, and by the largest margin with the tightest variance.

### What the `gated` soma is actually doing (gate ablation, 3 seeds, full eval)

Prompted by the user's hypothesis: maybe `gated` ties `sum` because the learned
gate is nearly uniform, and gating would pay off on a harder task. Testable
without retraining, because gated's two limits ARE the other two somas:
`beta -> 0` is the uniform mean, `beta -> inf` is `max`. So take the TRAINED
gated checkpoints and override only the pooling at evaluation
(`bdh_cl/scripts/gate_analysis.py`, cluster job 4309027).

| override | replay avg | delta | naive avg | delta |
|---|---|---|---|---|
| trained | 0.949 ± 0.013 | -- | 0.333 ± 0.000 | -- |
| uniform (beta->0) | 0.518 ± 0.156 | **-0.431** | 0.270 ± 0.050 | -0.063 |
| uniform, scale-matched | 0.626 ± 0.018 | **-0.323** | 0.309 ± 0.023 | -0.024 |
| hard (beta->inf) | 0.153 ± 0.024 | -0.796 | 0.122 ± 0.091 | -0.211 |
| beta x 0.25 | 0.786 ± 0.036 | -0.163 | 0.321 ± 0.013 | -0.012 |
| beta x 4 | 0.587 ± 0.107 | -0.362 | 0.279 ± 0.013 | -0.054 |
| beta x 16 | 0.191 ± 0.048 | -0.758 | 0.142 ± 0.084 | -0.191 |

Three findings, and they do not all point the same way.

**1. The gate is load-bearing, and not merely as a gain.** Removing it costs
0.431. Because `sum_c w_c a_c ~= mean_c(a) + beta * Var_c(a)`, dropping the gate
also shrinks every activation, so the bare `uniform` ablation confounds lost
structure with lost gain. The `uniform, scale-matched` row is the control: a
per-(head, neuron) gain fitted on the scored task itself, i.e. as generous to
the ablation as possible. It recovers only 0.108 of the 0.431. **About 75% of
the drop is structural**, so the input-dependent part of the gate is doing real
work. The trained beta is also at a genuine optimum -- perturbing it in EITHER
direction costs 0.16-0.76.

**2. But beta never trained.** `softplus(0) = 0.6931` is the initialisation.
After 2,400 steps the median beta is **0.6909** (replay) and **0.6904**
(naive), range 0.63-0.81. The gate weights stay near uniform
(`mean max_w = 0.3031` against 0.25 uniform, `eff_branches = 3.88/4`). The
network did not learn to route; it adapted its branch weights around the
pooling shape it was handed at init. The gate is a useful *fixed
nonlinearity*, not a learned router.

This resolves `gated ~= sum` without contradiction, by separating two questions
the earlier note ran together:

* does a trained gated net depend on its gate? **Yes**, strongly (-0.431).
* does choosing gated over sum at training time buy anything? **No**
  (+0.009, inside seed noise).

Training adapts to whichever pooling shape it is given, so the *choice* barely
matters -- as long as it is not `max`, which is bad both post-hoc (0.153) and
trained-from-scratch (0.868). That is consistent with the expressivity reading
above and not with the routing one.

**3. The gate is completely task-blind.** `mean max_w` per task, seed 0:
0.3042 / 0.3044 / 0.3031 (copy / reverse / sort) -- identical to three
decimals. So the branch specialisation reported in the previous section
(`same_branch_frac` falling 0.08 under replay) comes from the **branch
weights**, not from the gate. The routing mechanism is not what produces the
specialisation.

Consequence for the "harder task" hypothesis: difficulty is not the missing
ingredient, **within-task input heterogeneity** is. Every example of `copy`
wants the same operation, so the optimal per-neuron gate is constant and beta
has no gradient pressure -- which is exactly what the measurement shows. The
falsifiable version is a task whose required operation varies per example (e.g.
an op-code byte selecting copy/reverse/sort inside one task); the diagnostic is
then whether beta moves off 0.6931, which reads the mechanism directly instead
of inferring it from a 0.009 accuracy gap. NOT RUN.

### Caveat discovered while doing the above: exact_match has decode noise

Re-running the SAME checkpoint through the SAME quick eval gives
0.958 / 0.917 / 0.958 / 0.927. Decoding is greedy `argmax` and the eval set is
fixed, so this is float non-determinism in the forward pass amplified by
free-running generation: one flipped byte derails the whole continuation. On a
32-example subset that is +-0.04. It inflates apparent seed std on any
exact_match number, and it means single-seed quick-eval reads are not
trustworthy to better than ~0.05. Everything reported above is full eval
(100 examples) on GPU.

## PHASE 3: split MNIST (exp5) -- the effect replicates but shrinks 3x

Step 2C, requested by the user. 5 tasks, 14x14 images rendered as hex
characters, **class-incremental** (all tasks share one tag, so nothing at test
time identifies the task and the model may emit any digit). Note this is a
HARDER setting than exp2/exp4, where each synthetic task carries its own tag
and the runs were therefore task-AWARE.

Learnability was calibrated first (job 4310166) rather than assumed: trained on
one pair alone, both models reach held-out 1.000 on 0/1 and 0.995 on 8/9 by
step 250-500. `steps_per_task=800` was set from that, not guessed. Without this
check the smoke run's ~0.5 would have looked like a low ceiling.

Full sweep, 6 models x 3 strategies x 3 seeds = 54 runs, all complete:

| model | naive | replay | joint |
|---|---|---|---|
| bdh_den_gated | 0.198 | **0.866 ± 0.005** | 0.970 |
| bdh_den_max | 0.196 | 0.864 ± 0.019 | 0.982 |
| bdh_den_sum | 0.198 | 0.863 ± 0.005 | 0.973 |
| bdh (baseline) | 0.197 | 0.831 ± 0.030 | 0.976 |
| bdh_narrow (control) | 0.198 | 0.830 ± 0.021 | 0.970 |
| gpt (matched) | 0.194 | 0.782 ± 0.020 | 0.965 |

Controls behave correctly: naive is 0.197 for all six -- exactly the 1/5
class-IL floor -- and joint is 0.965-0.982 for all six, so the ceiling is intact.

**Three differences from exp4, two of which weaken the earlier framing.**

1. **The gain shrinks 3x.** Best dendritic over baseline is +0.035 here vs
   +0.118 in exp4. Still consistent (dendritic std ±0.005), but small.
2. **`bdh_narrow` now TIES baseline** (0.830 vs 0.831), where in exp4 it was
   clearly worse (0.764 vs 0.814). A model with 1/3 the parameters and 1/3 the
   neurons matching the full one means **capacity is not the binding constraint
   on this benchmark**, so "branches are a better use of parameters than width"
   does NOT hold here. It also suggests part of the exp4 result was a capacity
   story that easier-per-task real data removes.
3. **Soma choice stops mattering.** All three within 0.003, well inside seed
   noise, vs exp4's gated-over-max gap of 0.064. Consistent with the gate
   ablation: once training adapts to the pooling shape, which shape barely
   matters. The exp4 soma ORDERING should be treated as not replicated.

BDH beats matched GPT again (0.831 vs 0.782), replicating exp2's direction.

**Correction to an interim reading.** At n=1-2 seeds `max` looked best on this
benchmark (0.875) and I reported that. With all 3 seeds it is 0.864, tied with
the others. That is the third time a <3-seed reading has moved -- the rule in
the corrections list holds, including for my own interim notes.

## PHASE 4: synaptic consolidation without replay (exp6) -- NEGATIVE

The user's idea: accumulate per-synapse importance over a task, convert it to
neuron importance, and use it to stop later tasks overwriting those parameters.
Motivation is sound -- under naive training all six models sit at exactly the
1/5 class-IL floor (0.197) because nothing in the loss rewards leaving an old
parameter alone, and replay supplies that pressure only implicitly.

Implemented in `bdh_cl/cl/consolidation.py`, at BRANCH granularity (nh, k, N),
which is the finer unit only the dendritic model offers. Requires task
BOUNDARIES (importance is estimated at task end) but never task IDS.

### Three corrections to the idea as proposed

1. **There is no persistent synapse to read.** Upstream materialises attention
   as `scores = (QR @ KR.mT).tril(-1)`, a T x T TOKEN matrix, rebuilt every
   forward pass and discarded (`bdh.py:73`). Nothing decays across a task
   leaving only the important ones. The accumulator has to be constructed; the
   faithful version reconstructs `S = sum_t KR[t]^T (x) V[t]` (N x D per head),
   which upstream never forms because `tril(-1)` blocks the reassociation.
2. **The synaptic signal cannot be branch-granular.** Attention runs on the
   POST-soma code, so branches are already pooled away by the time the state
   exists. Asserted in `test_synaptic_signal_is_uniform_across_branches`. Only
   the activity signal varies across branches.
3. **Gradient scaling cannot work under AdamW.** The update is
   `lr * m_hat / sqrt(v_hat)`, so scaling every gradient by c scales m by c and
   sqrt(v) by c and CANCELS. Verified twice: as a unit test
   (`test_adam_cancels_a_constant_gradient_rescale`, a 1000x smaller gradient
   moves the weights the same distance to 1e-4) and by experiment below.

### Result: seven arms, all at the floor

Split MNIST, class-incremental, 5 tasks, `bdh_den_gated`, 1 seed, no replay:

| enforcement | lam | final | per-task retention |
|---|---|---|---|
| penalty | 1 | 0.196 | 0.00 0.00 0.00 0.00 0.98 |
| penalty | 10 | 0.199 | 0.00 0.00 0.00 0.00 0.99 |
| penalty | 100 | 0.197 | 0.00 0.00 0.00 0.00 0.98 |
| penalty | 1000 | 0.199 | 0.00 0.00 0.00 0.00 0.99 |
| penalty | 10000 | 0.198 | 0.00 0.00 0.00 0.01 0.97 |
| scale (control) | 1000 | 0.197 | 0.00 0.00 0.00 0.00 0.98 |
| mask, hard freeze | -- | 0.198 | 0.00 0.00 0.00 0.00 0.99 |

Naive floor is 0.197, replay is 0.866, joint 0.970. **Nothing moved.** The
penalty was live, not inert: `penalty_mean` per task rose 0.020 -> 0.093 with
lam, is exactly 0 on task 0 as it must be, and at lam=100 EXCEEDED the data
loss (0.058 vs 0.01). Displacement from the anchor provably shrank as lam grew
(penalty grew sub-linearly in lam). Plasticity was never the limiting factor
either -- the newest task still reached ~0.00 training loss at every lam.

### Why: two stacked failures, separated by diagnostic

`bdh_cl/scripts/diag_label_bias.py` on the exp5 naive checkpoint, per task,
free choice vs restricted to that task's own two classes:

| task | free choice | restricted to own 2 classes |
|---|---|---|
| mnist_01 | 0.000 | 0.500 |
| mnist_23 | 0.000 | 0.480 |
| mnist_45 | 0.000 | 0.570 |
| mnist_67 | 0.000 | 0.560 |
| mnist_89 | 0.995 | 0.995 |

1. **The readout collapses.** Free choice is 0.000: the model emits only the
   last task's two labels. No amount of FEATURE consolidation can fix this, so
   the reported metric cannot move even if features were perfectly preserved.
2. **The features are gone too.** Restricted accuracy is 0.48-0.57 and chance
   for two classes is 0.50. Handed a perfect readout the representation still
   carries nothing about old tasks.

So the benchmark's dominant failure is one consolidation does not address, AND
the features really are being destroyed. The `mask` arm is the sharpest
evidence about the signal: hard-freezing the most active branches still leaves
old tasks at exactly 0.00, so mean activation does not identify the parameters
that carry a task. That is the known weakness of activity-based importance --
EWC uses loss curvature and MAS uses output sensitivity for this reason.

### What this does and does not establish

Establishes: activity-based importance, at branch or neuron granularity, under
any of three enforcements and four decades of strength, does not protect
features on class-incremental split MNIST; and free-choice accuracy on this
benchmark cannot evaluate feature protection at all.

Does NOT establish that the idea is wrong. Untested, all NOT RUN:

* **Task-restricted accuracy as the metric.** The measurement that would
  isolate feature protection from readout bias. Without it the mechanism is
  being judged through a metric that is saturated by a different failure.
* **A sensitivity-based importance signal** (Fisher / output-gradient) instead
  of mean activation.
* **Hard mask without the 0.25 cap**, i.e. protecting more than 25% per task.
* **Consolidation combined with a readout fix** (class-balanced logits or bias
  correction), which is what the diagnostic says is required on this benchmark.
* The 3-seed `exp6_consol.yaml` sweep, including `replay+consol`. Not queued:
  the 1-seed calibration showed nothing to measure, so spending 24 GPU runs on
  it would have been waste.

## PHASE 5: prototype (nearest-class-mean) readout -- a second, independent metric

The user's proposal, and it is better than the task-restricted accuracy I had
suggested: classify by nearest class prototype instead of the learned head. No
output layer to collapse onto recent labels, and -- unlike task-restricted
accuracy -- it uses NO task id at test time, so it stays genuinely
class-incremental. Standard strong baseline (iCaRL's NCM, SLDA).

Run on exp5's existing checkpoints, so zero retraining: 6 models x 3 strategies
x 3 seeds, eval cost only (`bdh_cl/scripts/proto_eval.py`, job 4313016).

Two protocols, and the DIFFERENCE is the measurement:

* `frozen` -- each class's prototype computed at the end of ITS OWN task, from
  that task's training data, never touched again. What an online exemplar-free
  method could actually store; a class mean is not stored examples, so this does
  not smuggle in replay. **This is the honest number.**
* `refreshed` -- all prototypes recomputed with the FINAL model. Needs old data,
  so it is an ORACLE, not a method. Upper bound only.
* `refreshed - frozen` = **representation drift**: how far features moved out
  from under the stored prototypes. This is the quantity a consolidation
  mechanism is supposed to shrink, so it is the instrument for testing one.

### The instrument validates itself

Under `joint` drift is **0.000 ± 0.002** for every model. Joint trains all tasks
at once, so there is no sequential drift and frozen must equal refreshed. It
does, to three decimals. Prototypes also reach 0.978-0.982 there, so the readout
itself is not the bottleneck.

### Results (3 seeds)

| strategy | model | frozen | refreshed (oracle) | drift | generative |
|---|---|---|---|---|---|
| naive | bdh_den_gated | 0.201±0.001 | 0.390±0.056 | +0.188 | 0.198 |
| naive | bdh | 0.200±0.002 | 0.411±0.065 | +0.211 | 0.197 |
| naive | gpt | 0.196±0.002 | 0.452±0.038 | +0.256 | 0.194 |
| replay | bdh_den_gated | 0.610±0.066 | **0.912±0.013** | +0.302 | 0.866 |
| replay | bdh_den_max | 0.614±0.051 | 0.912±0.013 | +0.298 | 0.864 |
| replay | bdh_den_sum | 0.545±0.065 | 0.891±0.016 | +0.346 | 0.863 |
| replay | bdh | 0.560±0.092 | **0.909±0.030** | +0.350 | 0.831 |
| replay | bdh_narrow | 0.582±0.093 | 0.886±0.026 | +0.304 | 0.830 |
| replay | gpt | 0.527±0.025 | 0.868±0.010 | +0.342 | 0.782 |
| joint | all | 0.978-0.982 | 0.978-0.982 | 0.000 | 0.965-0.976 |

### Four findings

**1. CORRECTION to the phase-4 write-up.** I claimed free-choice accuracy
"cannot evaluate feature protection at all" because readout collapse saturates
it. Too strong. Under naive, frozen-prototype accuracy is 0.199-0.201 against
the generative 0.194-0.198 -- the two metrics AGREE. The learned head was not
hiding a healthy representation; the features genuinely are destroyed. The
prototype readout earns its place under REPLAY, not under naive.

**2. BDH > GPT replicates under an independent readout.** Representation quality
(`refreshed`) under replay: BDH 0.909 ± 0.030 vs GPT 0.868 ± 0.010, matching the
generative ordering (0.831 vs 0.782). Two readouts that share no output
parameters agree, which makes this the best-supported claim in the project.

**3. The dendritic advantage does NOT replicate under this readout.** Under
replay, `refreshed` is 0.912 ± 0.013 for gated and **0.909 ± 0.030 for baseline
BDH** -- identical. Yet the generative metric separates them (0.866 vs 0.831).
So the dendritic gain is NOT explained by a better representation; whatever it
buys happens in the interaction with the learned head. This is a real caveat on
the phase-2/3 claim and it should be reported: branches improve the end-to-end
metric without measurably improving the features.

**4. Drift is large for everyone** (~0.30 under replay) and roughly
model-independent within error bars. So a drift-reduction mechanism has genuine
headroom to demonstrate itself -- this is the metric consolidation should be
tested against, not free-choice accuracy.

### An honest caveat against prototypes as a classifier

Under replay the LEARNED head beats frozen prototypes for every model
(0.78-0.87 vs 0.53-0.61), because replay keeps retraining the head on old
labels. Prototypes are the right INSTRUMENT for isolating drift; they are not
automatically a better classifier in a setting where the head still sees old
classes.

### Consequence for testing consolidation

The target is now measurable without a task id: does consolidation raise
`frozen` under naive above 0.20 (chance is 0.10 over 10 classes), and does it
shrink drift below ~0.19? NOT RUN.

## PHASE 6: winner-take-all (exp7) -- neuron-level WTA HURTS

The user asked to try winner-take-all with the dendrites. Branch-level WTA is
already `soma: max` and already measured twice, so exp7 tests the two forms that
were NOT: winner-take-MOST across branches (`soma: topk`), and k-WTA ACROSS
NEURONS (`wta_frac`, keep the top fraction of the layer's units per token).
Neither adds a parameter, so all arms stay matched.

Split MNIST, class-incremental, 5 tasks, 3 seeds:

| arm | replay | naive |
|---|---|---|
| bdh_den_max (branch WTA) | **0.865 ± 0.029** | 0.199 |
| bdh_den_gated | 0.840 ± 0.009 | 0.196 |
| bdh_den_topk2 (2 of 4 branches) | 0.815 (n=1, see below) | 0.197 (n=2) |
| bdh_den_max_wta10 (+top 10% neurons) | 0.779 ± 0.017 | 0.198 |
| bdh_den_sum_wta05 (neuron WTA only) | 0.779 ± 0.061 | 0.197 |
| bdh_den_max_wta05 (+top 5% neurons) | **0.731 ± 0.051** | 0.197 |

**k-WTA across neurons hurts, monotonically in sparsity**: 0.865 -> 0.779 (10%)
-> 0.731 (5%). It does not lift the naive floor either -- every arm is
0.196-0.199 there, as every arm of every experiment has been.

**Undertraining is ruled out.** Single-task ceilings (fresh model, each task
alone) are ~0.99 for every sparse arm: max 0.991, max_wta05 0.988, max_wta10
0.989, sum_wta05 0.993/0.990. Plasticity is intact, so the drop is genuine
forgetting rather than a capacity artifact. Without this control the result
would have been indistinguishable from "the sparse model just could not learn".

**Winner-take-most is not better than winner-take-all**: topk2 0.815 vs max
0.865. Only 1 replay seed so far (below), so held loosely.

### Why, probably: sparsity COMPOUNDS through the multiplicative gate

BDH's output is the PRODUCT `x_sparse * y_sparse`, and exp7 applies `wta_frac`
to both paths. Two 5%-dense codes give a ~0.25%-dense conjunction, against
~0.74 unconstrained -- roughly a 300x reduction in active conjunctions. That is
an architecture-specific cost with no analogue in the feedforward MLPs the
sparse-coding results (Ahmad & Scheinkman; Numenta) come from. Consistent with
`sum_wta05` (0.779) beating `max_wta05` (0.731): removing one of the two
competitions helps.

exp8 (job 4315135) tests it directly by constraining ONE path, which puts
conjunction density at ~5% instead of ~0.25%. Launched only after exp7
confirmed the effect at 3 seeds.

### A note on cost

`torch.topk` along the branch axis is far more expensive than `amax`/`sum`:
array task 2 hit the 2h limit at 2 of 3 seeds while every other arm finished in
25-42 min. The missing seed is rerunning with a 4h limit (job 4315130) rather
than being reported as a 2-seed result.

## PHASE 6b: why neuron WTA hurts (exp8) -- two stacked costs, not one

exp7 applied `wta_frac` to BOTH latent paths. Since BDH's output is the product
`x_sparse * y_sparse`, two 5%-dense codes give a ~0.25%-dense conjunction, so the
predicted cause was multiplicative compounding. exp8 tests it by constraining
ONE path. Split MNIST, class-incremental, replay, 3 seeds:

| neuron WTA applied to | final | damage vs none |
|---|---|---|
| nothing | 0.864 ± 0.022 | -- |
| y path only | 0.834 ± 0.028 | 0.030 |
| x path only | 0.792 ± 0.013 | 0.072 |
| both paths | 0.740 ± 0.065 | 0.124 |

**Compounding is confirmed**: constraining both is worse than constraining
either alone. And the two costs are roughly ADDITIVE -- 0.030 + 0.072 = 0.102
against a measured 0.124 for both -- so slightly super-additive, i.e. a modest
multiplicative interaction on top of two independent path costs.

**The asymmetry was not predicted and has a clean architectural cause.**
Constraining `x` costs 0.072, more than twice the 0.030 for `y`. In the forward
pass `x_sparse` does DOUBLE DUTY: it is both Q and K in
`core.attn(Q=x_sparse, K=x_sparse, V=x)` **and** a factor in the multiplicative
gate. `y_sparse` appears only in the gate. So sparsifying x damages attention
AND the gate; sparsifying y damages only the gate.

So the exp7 explanation was incomplete rather than wrong: there are two costs
stacked, and the larger one is about which path carries more structural load.
This is a second, independent reason to ablate the product -- the x path's dual
role is a candidate mechanism for BDH's advantage over the matched Transformer
in its own right.

## PHASE 7 / MILESTONE 1: persistent synaptic memory, ungated -- NEGATIVE, and
## the negative is the argument for gating

`Sigma <- lambda Sigma + eta x^T v`, read as `x @ Sigma`, position-free, Hebbian
(a buffer, never a parameter), causal (read the state from before the batch).
No dendrites, no context gating, no task signal: milestone 1 asks only whether a
decaying outer-product state that survives sequence boundaries buys anything
when an environment RECURS.

Schedule A -> B -> A with 5-way environments ({0..4} vs {5..9}), 3 seeds.
Headline is savings, not final accuracy: `1 - steps_2/steps_1` to reach 80% of
that environment's own fresh-model ceiling.

| arm | v1 steps | v2 steps | savings | final acc | retention at return |
|---|---|---|---|---|---|
| **bdh (no memory)** | **142** | **42** | **0.711** | **0.660 ± 0.006** | 0.000 |
| mem_fast (decay 0.99) | 242 | 67 | 0.704 | 0.650 ± 0.003 | 0.000 |
| mem_slow (decay 0.999) | 283 | 133 | 0.562 | 0.626 ± 0.013 | 0.000 |
| mem_nodecay (decay 1.0) | 242 | 267 | -0.107 | 0.598 ± 0.022 | 0.000 |

**Three findings.**

1. **The no-memory control is best on every measure** -- fastest first visit
   (142), fastest revisit (42), highest savings (0.711), highest final accuracy.
   Savings of ~0.7 exist WITHOUT any persistent state, because a network that
   already knows the byte format and output structure relearns faster anyway.
   The bar was never zero, it was the control. Without that arm, mem_fast's
   0.704 would have read as a success.
2. **Harm is monotone in how much the state retains**: final accuracy 0.650
   (decay 0.99) -> 0.626 (0.999) -> 0.598 (1.0), and savings 0.704 -> 0.562 ->
   -0.107. The more the memory persists, the worse the model does.
3. **Retention at return is exactly 0.000 for every arm**, memory included. The
   slow state preserves nothing about the absent environment.

**Why this is the argument for the next step rather than against the idea.**
An UNCONDITIONAL memory is a single sum over all past experience. When the
environment changes, what it recalls is wrong, so it does not merely fail to
help -- it actively misleads, and it misleads more the longer it remembers.
That is exactly finding 2. Which is the case for context-gated plasticity:
`Delta Sigma^(k) = g_k(c_t) x^T v` writes and reads per-context slots, so a
recall from environment B does not contaminate environment A. Milestone 1
establishes that the interference is real and that gating is necessary rather
than an optional refinement -- and if gating had been built first, its benefit
could not have been attributed.

**A trap in the savings ratio, worth recording.** mem_slow shows 0.562 savings,
which looks respectable, but it gets there from 283 -> 133 while the control
goes 142 -> 42. A ratio flatters an arm whose first visit was slow. Absolute
steps must be reported alongside it, and are.

**Caveats / NOT RUN.** One `eta` (1.0) and one `read_scale` (1.0) throughout, so
the honest claim is that THIS parameterisation is harmful, not that no
parameterisation could be neutral -- the monotone harm suggests the write may
simply be too large, and a smaller `eta` might be inert rather than damaging.
Also: two environments, one revisit, and no continuous drift. The measurement
criterion matters too: steps-to-target requires two consecutive samples above
target, because the quick-eval curves are non-monotone under the ~0.04 decode
noise; the bare first-crossing variant moves the control's savings from 0.711 to
0.500, so both are computable via `--raw`.

**An earlier version of this experiment was unmeasurable** and is kept as
`exp9_binary_saturated`: with binary-pair environments every arm hit target at
the FIRST sampled step, so savings could not come out positive by construction.
5-way environments were the fix.

## PHASE 8: dendritic BDH on Shakespeare at the ORIGINAL config (exp10) -- POSITIVE

Requested by the user: train the dendritic model on the dataset and config
upstream reported, to see what it gains over baseline BDH. Same script that
produced the baseline and GPT numbers, so it is the same code path.

Held identical to upstream: 6 layers, d=256, 4 heads, dropout 0.1, block 512,
batch 32, 3000 steps, AdamW 1e-3 / wd 0.1, no schedule, no clipping, fp32, same
corpus and 90/10 split. Two things change, both necessarily:
`neuron_unit` (the thing under test) and `mlp_internal_dim_multiplier` 128 -> 43.
The multiplier MUST move because the two encoders scale with k; holding it fixed
would hand the dendritic model ~3x the parameters and measure capacity instead
of architecture. 43 puts the total at 25.49M vs upstream's 25.30M (+0.78%).

3 seeds each:

| model | params | train bpb | HELD-OUT bpb | tok/s |
|---|---|---|---|---|
| bdh | 25,296,896 | 0.1605 ± 0.0006 | 4.5751 ± 0.0306 | 14k |
| **bdh_den_sum** | 25,493,504 | 0.2054 ± 0.0050 | **4.3338 ± 0.0414** | **24k** |
| bdh_den_max | 25,493,504 | 0.2019 ± 0.0082 | 4.4351 ± 0.1143 | 20k |
| gpt (matched) | 25,436,672 | 1.0727 ± 0.1083 | 2.6396 ± 0.1260 | 148k |

**1. Held-out bpb improves by 0.241** (4.575 -> 4.334), seed spreads ±0.031 and
±0.041, non-overlapping by roughly 6 sigma. The cleanest positive for dendrites
in this project, and the first OUTSIDE continual learning.

**2. It memorises less while generalising better.** Train bpb goes UP
(0.161 -> 0.205) while held-out goes DOWN. Opposite directions is the signature
of reduced overfitting rather than a shifted operating point -- which matters,
because BDH's severe overfitting at this config (train 0.16 vs held-out 4.58)
was one of the phase-1 findings.

**3. It is also 1.7x FASTER** (24k vs 14k tok/s) at matched parameters. Not a
free lunch that needed arranging: BDH's attention costs O(T^2 * N) in the neuron
count, and parameter matching forces the dendritic model to N' = 2752 rather
than 8192, so the dominant term is ~3x cheaper. Better generalisation AND a
speedup at equal parameters is an unusual combination and worth stating.

**4. `sum` beats `max`** (4.334 vs 4.435, and max's variance is 3x larger).
`sum` has no routing and no branch specialisation, so this is the fourth
independent result pointing at per-neuron EXPRESSIVITY rather than routing:
alongside beta never training, the soma dead heat on MNIST, and the prototype
readout showing identical representation quality.

**Caveats.**

* Dendrites do NOT close the gap to the matched Transformer: GPT is still 1.69
  bpb better. 0.24 of the 1.94 gap recovered, about 12%.
* `gated` is ABSENT because it OOMs at this config -- 93.1 GiB is not enough; it
  materialises the branch softmax alongside the branch activations. Measured,
  job 4320011: bdh 69.3 GiB, sum 70.2, max 76.1, gated OOM. Running it would
  need a batch-size deviation, so it was left out rather than silently changing
  the config. The variant that won on synthetic CL is therefore untested here.
* One corpus (1.1MB) at ~44 epochs, so all of these models are overfitting by
  construction; this says dendrites overfit LESS, not that they generalise well.

**Scheduling note, recorded because it cost a day.** The first submission asked
`--time=04:00:00` for a run that takes 36 minutes. On a cluster with most nodes
drained for maintenance, a 4h request cannot be backfilled into the gaps a 1h
request fits, and five of six tasks sat PENDING for ~24h. Resubmitted at 1h they
all ran. Also relevant: the account's fairshare is now exhausted
(`EffectvUsage 1.000000`, FairShare 0.226) from this project's job volume, so
future sweeps should be sized tightly.

## PHASE 9: depth control on Shakespeare (exp13) -- the dendritic claim does not survive

`bdh_deep`: relu(relu(x @ A) @ B), same parameters (25.50M) and same neuron count
(N' = 2752) as the dendritic model, extra capacity spent on one hidden layer with a
learned projection instead of k branches with fixed pooling. Upstream config, 3 seeds.

| model | train bpb | held-out bpb | per-seed held-out |
|---|---|---|---|
| bdh | 0.161 ± 0.001 | 4.575 ± 0.031 | 4.580 / 4.603 / 4.542 |
| bdh_den_sum | 0.205 ± 0.005 | 4.334 ± 0.041 | 4.348 / 4.287 / 4.367 |
| **bdh_deep** | 0.360 ± 0.056 | **3.992 ± 0.187** | 4.133 / 3.780 / 4.063 |

**The depth control BEATS the dendritic model on every seed.** Dendritic grouping
contributes nothing beyond plain per-neuron depth, and less than it.

**But the comparison itself is confounded, and that matters more.** Across all nine
BDH-family runs, correlation(train bpb, held-out bpb) = **r = -0.976**. The less a
model memorises, the better it scores held-out -- even within the depth arm, the
seed that memorises least (train 0.422) generalises best (3.780). On a 1.1 MB corpus
at ~44 epochs, held-out bpb is measuring how much a model OVERFITS, not how good its
architecture is. Both "wins" are most likely regularisation effects: both variants
replace 8,192 neurons with 2,752 and both memorise less.

**Correction to phase 8.** "Dendritic BDH improves held-out bpb by 0.24" stands as a
measurement but NOT as an architectural claim. It should not be reported as
evidence that dendrites improve language modelling.

What would settle it, NOT RUN: (1) bdh_narrow at N=2752 on Shakespeare -- if a
smaller plain model also "wins", it is pure capacity/regularisation; (2) any
comparison on a corpus large enough that no model sees it more than ~once.

The continual-learning depth control is the cleaner test, because that benchmark
samples fresh data every step and is not over-epoched. Rerun after a test-tolerance
fix (the flat 1% parameter-match tolerance failed at a toy shape where integer
rounding of the hidden width gives -1.56%; real configs match to 0.02% / 0.12%).

## PHASE 9b: depth control on CONTINUAL LEARNING -- dendrites win, and it reverses 9a's reading

Same exp13 control on the synthetic CL benchmark (3 tasks, 5% replay, 3 seeds), which
samples fresh data every step and so is NOT dominated by memorisation.

| model | final | forgetting | per-seed |
|---|---|---|---|
| bdh | 0.830 ± 0.050 | +0.167 | 0.793 / 0.810 / 0.887 |
| bdh_den_sum | **0.948 ± 0.022** | **+0.048** | 0.933 / 0.937 / 0.973 |
| bdh_den_gated | 0.940 ± 0.030 | +0.059 | 0.907 / 0.950 / 0.963 |
| bdh_deep | 0.787 ± 0.114 | +0.208 | 0.657 / 0.833 / 0.870 |

Every dendritic seed (min 0.907) beats every depth seed (max 0.870). The depth control
does not even beat baseline and forgets more. Naive is 0.333 for all, as always.
Dendritic arms replicate exp4 (0.940 / 0.948 vs 0.932 / 0.923).

**CORRECTION.** After 9a I wrote that dendritic grouping "adds nothing beyond plain
per-neuron depth" and that the dendrite line was closed. Wrong. On the unconfounded
benchmark, per-neuron branch grouping is specifically responsible for the retention
gain; extra nonlinear depth at equal parameters and equal neuron count does not
reproduce it. I had also predicted a tie. The Shakespeare ordering (deep > dendritic)
is the confounded one (r = -0.976 between train and held-out bpb).

**Candidate mechanism (hypothesis, NOT tested).** bdh_deep's hidden layer is SHARED:
every output neuron reads every hidden unit, so an update learned on one task changes
the input to every neuron -- non-local by construction. A dendritic branch is PRIVATE
to one neuron, so an update to it touches only that neuron. This is LSL's locality
argument (Ben-Kish et al. 2026) applied to parameter sharing rather than to inputs. It
predicts the benefit comes from structure, not routing -- consistent with `sum` matching
`gated` and beta never training. Testable with a cross-task interference probe
comparing the two units.

## PHASE 10: cross-task interference probe (exp17) -- the locality hypothesis FAILS

Hypothesis (mine): BDH's sparse conjunction code (locality across inputs) and private
dendritic branches (locality across parameters) make weight updates local, which is why
BDH and dendritic BDH forget less. Probe: train on A, take 1/5/20/50 optimizer steps on
B, measure damage to A per unit of B learned, gradient alignment, how concentrated the
change to A's output-layer code is, and A-vs-B code overlap. exp13 models and spec;
3 seeds x 2 task pairs.

| model | damage ratio @50 | grad cos | frac units w/ 90% of change | A-B overlap | A-A overlap |
|---|---|---|---|---|---|
| bdh | 0.941 ± 0.324 | -0.002 | 0.362 | 0.123 | 0.127 |
| den_sum | 0.953 ± 0.521 | -0.003 | 0.453 | 0.194 | 0.190 |
| den_gated | 0.931 ± 0.475 | -0.008 | 0.453 | 0.201 | 0.196 |
| deep | 0.881 ± 0.253 | +0.005 | 0.245 | 0.179 | 0.178 |
| gpt | 0.734 ± 0.154 | -0.031 | 0.790 | 0.085 | 0.145 |

All three predictions fail. Damage ratios do not separate (GPT lowest), so the probe
cannot distinguish dendritic from depth despite exp13's 0.948 vs 0.787 retention gap.
The depth control's change is the MOST concentrated, dendritic LESS than baseline --
the private-branch story is contradicted. BDH's codes overlap MORE across tasks than
GPT's, and barely separate tasks at all (cross ~= within).

Reading: this probe has no replay, and without replay every model sits at the same
naive floor. The dendritic advantage only ever appears UNDER replay, so branches
probably make replay more effective rather than making updates intrinsically local.
Next test: replay-fraction sweep (0.5/1/2/5%) -- if branches improve replay efficiency,
their advantage should be largest at small buffers. NOT RUN.

## PHASE 1 FINAL: sparsity at the ORIGINAL 25M config (3 seeds each)

Density probed on the **held-out** split, matched on optimizer step:

| step | BDH x_sparse | BDH y_sparse | GPT mlp_hidden |
|---|---|---|---|
| 0 | 49.8 ± 0.3 | 49.7 ± 0.2 | 49.8 ± 0.5 |
| 1000 | 24.1 ± 0.4 | 25.2 ± 1.0 | 21.0 ± 1.0 |
| 2999 | **21.9 ± 0.1** | **18.3 ± 0.3** | **13.5 ± 0.7** |

Both sparsify hard from ~50%. GPT stays somewhat sparser. **The laptop-scale
claim that BDH is much denser (40% vs 30%) was mostly an under-training
artifact** -- at `tiny`, 800 steps on 1.6M params never reached convergence.
The direction survives; the magnitude does not.

Throughput on identical H100s: BDH 13.5k tok/s vs GPT 148k -- **~11x slower**
at matched parameters.

**BDH memorises far harder and generalises far worse.** Both models see
3000 x 32 x 512 = ~49M tokens over a 1.1MB corpus (~44 epochs), so train bpb
measures memorisation, not quality. With all 3 seeds now in:

| | train bpb | HELD-OUT bpb | gap |
|---|---|---|---|
| BDH | 0.1605 ± 0.0006 | **4.575 ± 0.031** | +4.415 |
| GPT (matched params) | 1.073 ± 0.108 | **2.640 ± 0.126** | +1.567 |

BDH fits the training text **6.7x better** and generalises **1.94 bits/byte
worse**. The seed spread is tiny (BDH ± 0.031) and the two are nowhere near
overlapping, so this is not a noise artifact. Reporting train bpb alone --
which an earlier version of this file did -- made BDH look like a 7x quality
win when the opposite is true on held-out data. That correction is recorded in
the corrections list below.

Caveat on scope: this is one corpus (Shakespeare, 1.1MB), one config
(upstream's own defaults), and heavily over-epoched by construction. It says
BDH overfits *this* setup much harder than a parameter-matched GPT; it does not
establish a general generalisation gap, which would need a corpus large enough
that neither model can memorise it. NOT RUN.

This is worth holding next to the continual-learning result rather than
separately: BDH's CL advantage coexists with much heavier memorisation, and the
high representational overlap (Jaccard 0.61-0.85 vs GPT's 0.20-0.37) is
consistent with a model that packs tasks into a shared, heavily-reused code.
Whether the CL gain and the overfitting share a cause is untested.

## Other results

**Length generalization essentially fails.** On inputs longer than any seen in
training, the final naive model scores 0.00 / 0.00 / 0.07 on copy / reverse /
sort (replay: 0.00 / 0.00 / 0.12). Both models learn length-specific patterns
rather than the algorithm. Reported rather than buried.

**Plasticity is under-resolved, not null.** After the budget cut,
`eval_every=400` over 800 steps leaves only 3 curve points, so steps-to-target
can only take the values 0 / 400 / 799 / None -- every task reports 400/400.
That is a sampling-resolution artifact. At `eval_every=150` in the earlier
5-task run the sequential model visibly learned each new task *faster* than a
fresh one (reverse at step 300: 0.688 sequential vs 0.391 fresh). Plasticity
should be reported as under-resolved at this setting, not as "no effect".

## Open / not yet done

* Step 2A: data module built and tested; **no language CL run yet**.
* Step 2C MNIST, Step 4 in-context: not started.
* Non-Latin language stream: dropped, see above.
* Compute-matched (as opposed to parameter-matched) comparison: not run.
* "Keep optimizer state" variant: flag exists, not run.
* `small` and `base` presets: not validated by a real run (see above).


## Depth control recovered from cluster — 1 October 2026

The three exp13 Shakespeare runs completed remotely but were absent locally. Recovered JSONs are in `bdh_cl/results/cluster/exp13_recovered`. Deep projection scores 3.992 held-out bits/byte (three seeds), versus sum dendrites 4.334 and baseline BDH 4.575, at near-identical deep/sum parameter counts and similar throughput. This weakens a dendrite-specific interpretation of the language gain; it does not establish the mechanism or resolve continual learning.

The previous CL task aborted at an impossible blanket 1% parameter-matching assertion for a toy integer-rounded width. Repaired the rounding-bound assertion while retaining <1% matching for production shapes; local neuron tests: 38 passed. Submitted CL-only array 4328965 (seeds 0–2), one GPU and one-hour limit per seed; initial status pending. See the recovered REPORT.md and `slurm/exp13_depth_repair.sbatch`. No completed CL results yet.

## Branch-indexed persistent-state hypothesis — tested 1 October 2026

Implemented in `bdh_cl/models/compartment_bdh.py`, with prompt-only classification runners and five mechanism tests. Full BDH remains trainable; four bank slots per head preserve branch identity through memory access. This is a bank-level prototype, not one independent context gate per neuron branch. All arms have 1,688,960 parameters; memory arms have 720,896 bytes of slow state. Shared and uniformly accessed bank controls are algebraically equivalent and produced identical results.

The raw-read smoke signal was negligible (<0.01% of fast attention); normalized reads were therefore used for the actual pilot and confirmation runs, with magnitude capped relative to fast attention at read_scale=0.1. The 300-step pilot learned both five-way environments above 90% but showed zero old-class retention. The full experiments use 800 updates per stage, three seeds, no replay or task IDs, and A → B → A.

All three seeds of disjoint-class recurrence (job array 4330684) had zero A accuracy before relearning in every arm. New B learning was about 97–98%. In same-label spatial-permutation recurrence (array 4330763), pre-update A retention was 38.5% no memory, 45.0% shared/uniform, and 43.7% selective. Erasing state at identical learned weights raised the respective shared and selective means to 49.3% and 45.0%. This does not establish useful old-context retrieval from the stored compartments. A benefit from altered training dynamics is possible, not proven. Per-head aggregate routing on disjoint A/B was similar and selected the same dominant bank in all seed/head combinations; these aggregate probes do not prove all input-level routing is identical.

Both arrays and pilot 4330542 completed successfully. The combined pilot/confirmation GPU allocation elapsed time was about 31 minutes. Reports and raw metrics: `bdh_cl/results/cluster/exp14_analysis/REPORT.md`, `exp14_analysis/summary.json`, `exp14_compartment`, and `exp14_compartment_domain`. Checkpoints remain on the cluster. Next isolate stable context addressing and compare against generic equally sized memory slots before adding capacity or claiming a neuron-local dendritic mechanism.

The exp13 three-seed repair comparison also completed: synthetic replay accuracy baseline 84.4%, gated 93.8%, sum 92.9%, deep 79.0%; all naive variants near 33.3%. Thus branches outperform the depth control on this synthetic replay benchmark despite the opposite ordering on Shakespeare. This supports a benchmark-specific architectural benefit, not replay-free memory protection.

## 2026-10-04: Exp16 visual pretraining implementation
Added patch-input visual BDH and fixed-budget grouped value-side residual branches with slow/fast Gaussian statistics. CIFAR-100 base-50 + five increments of 10 classes; one-seed pilot, no replay, all-seen-class evaluation. Shared pretrained checkpoint across full fine-tuning, frozen/trainable always-open, and frozen/trainable Gaussian arms. W&B group `exp16_cifar100_value_branches`. Detailed protocol: `bdh_cl/results/cluster/exp16_cifar100/PROTOCOL.md`. Unit test job 4369330: 4 passed, exit0. Training job identities/status recorded in protocol after submission.

## 2026-10-06 — Branch segregation: Step 1 diagnostic, Step 2 setup (exp18)

**Step 1 (from existing exp5 split-MNIST data, 3 seeds, no new runs).** This measures whether different digit-pair tasks use different dominant branches per neuron. The metric is same_branch_frac on the x stream, last layer only, init → final:

| Variant | Init | Naive | Replay | Joint |
|---|---|---|---|---|
| den_max | 0.990 | 0.971 | 0.949 | 0.942 |
| den_sum | 0.993 | 0.942 | 0.924 | 0.925 |
| den_gated | 0.992 | 0.938 | 0.922 | 0.930 |

The y stream is similar (0.94–0.96). Branches do **not** segregate by task, even under joint training. The precondition for branch-level protection (Cichon & Gan style) is therefore absent in dense dendritic BDH. Caveats: only the final layer is captured, and the inputs are byte-rendered MNIST.

**Step 2 (Limbacher & Legenstein 2020 ingredients).** New `sparse_dendritic` unit:
- a fixed contact budget per branch;
- SET-style rewiring (prune the weakest active synapse, regrow a random dormant one at zero; Adam state reset);
- max soma.

This is a simplified DEEP R: no sign constraint and no Langevin noise. The design is a 2×2 at matched neurons (1408) and matched synapses per neuron (64):
- k=1×64 contacts vs k=4×16 contacts;
- fixed mask vs rewired (every 50 steps, 10%).

References are dense den_max and bdh_narrow. Config: `bdh_cl/configs/exp18_sparse_rewire.yaml`; cluster job: `slurm/exp18_sparse_rewire.sbatch`.

- Tests: 12 new (`test_sparse_dendritic.py`); 60 passed together with the neuron-unit and equivalence tests.
- Smoke (`smoke_exp18`, 7×7, 60 steps/task, seed 0, naive, MPS): completed end to end in 7.9 min, and branch probes are populated for every sparse arm. Smoke same_branch_frac (x, 01|23) went init 0.969 → final 0.998 (rewire) and 0.994 (fixed). This was 300 steps, so it is not evidence either way.
- **The full 3-seed run has NOT been run yet.**

### exp18 results: rewiring does not produce branch segregation (job 4379585, capella CUDA, 3 seeds)

**Branch metric.** same_branch_frac, mean over all 10 task pairs, last layer, init → final, mean ± std over 3 seeds:

| Arm (k=4) | Naive x | Replay x | Replay y |
|---|---|---|---|
| dense den_max | 0.991 → 0.973±0.008 | 0.991 → 0.947±0.024 | 0.990 → 0.955±0.007 |
| sparse fixed (16 contacts) | 0.993 → 0.977±0.006 | 0.993 → 0.958±0.009 | 0.991 → 0.952±0.004 |
| sparse rewired | 0.993 → 0.971±0.002 | 0.993 → 0.950±0.003 | 0.991 → 0.952±0.005 |

The k=1 arms are 1.000 by construction (one branch).

**Replay final accuracy:**

| Arm | Accuracy |
|---|---|
| bdh_narrow | 0.823±0.037 |
| dense den_max | 0.859±0.019 |
| point fixed | 0.844±0.029 |
| point rewire | 0.824±0.028 |
| den fixed | 0.846±0.022 |
| den rewire | 0.861±0.007 |

Every naive arm sits at 0.196–0.200 final accuracy with forgetting ≥0.985, i.e. readout collapse.

**Verdict.** Sparse contacts plus SET-style rewiring plus max-soma competition do not make branches segregate by task. Rewired ≈ fixed ≈ dense, all about 95–97% shared dominant branch. On accuracy, the 2×2 shows no rewiring effect and no branch-vs-point effect beyond seed noise. The Limbacher ingredients alone, in this simplified gradient-trained form, are insufficient.

**Not done.**
- proto_eval on the exp18 checkpoints; they are on the cluster, not pulled.
- Any check of rewiring rate beyond the unit and runner-hook tests, which passed on the cluster before training.
- Rewiring was gradient-magnitude based (SET), not Hebbian or coactivity driven like Limbacher. The most plausible missing piece is that nothing correlates a branch's inputs with a task; a task-context signal on the branch gate is the next candidate.

## 2026-10-07 — Coactivity-driven rewiring, setup (exp19)

Direct follow-up to exp18's closing note: swap the SET regrow rule (prune
weakest active synapse, regrow a *random* dormant one) for a Hebbian/
coactivity-driven one, isolating the regrow-target rule as the only variable.

`SparseDendriticNeuronUnit` gets `rewire_mode: "random" | "coactivity"`
(default `"random"`, so exp18 stays byte-identical). In `"coactivity"` mode,
every training-mode forward pass accumulates an EMA (`coactivity_decay=0.95`
default) of `input_i * branch_activity_c` into a `(nh, D, k*N)` buffer —
the real-BDH analogue of the toy lab's clustered-rewiring trace
(`this.trace[u,i] = 0.9*trace + 0.1*g[u]*xEff[i]`). `rewire()` then regrows
at the dormant synapse with the highest trace instead of a random one; prune
rule, schedule, and budget are unchanged. The trace only updates in
`self.training` mode, so eval/probe forward passes do not pollute it.

Same 2x2 shape as exp18, with the SET-rewire arms replaced by coactivity
ones: `bdh_cl/configs/exp19_coactivity_rewire.yaml`; cluster job:
`slurm/exp19_coactivity_rewire.sbatch`.

- Tests: 5 new in `test_sparse_dendritic.py` (invalid `rewire_mode` rejected,
  trace stays zero outside coactivity mode, trace only accumulates in
  training not eval, rewire regrows the highest-trace dormant synapse,
  selectable by name). 225/225 passed across the full `bdh_cl` suite.
- Smoke (`smoke_exp19`, 7×7, 60 steps/task, seed 0, naive, MPS): completed
  end to end in 8.9 min (exp18's smoke was 7.9 min), all 6 arms ran, losses
  and exact-match curves look like exp18's. Smoke same_branch_frac (x,
  01|23) went init 0.954 → final 0.994 (coactivity rewire) vs. 0.954 → 0.995
  (fixed) — indistinguishable at 300 steps, same caveat as exp18's smoke.
- Full 3-seed run: launched with user go-ahead as capella job 4383716
  (array 0-2, wandb group `exp19_coactivity_rewire`). The in-job pytest
  passed 79/79, and the arrays COMPLETED in 38.6 / 40.8 / 37.2 min. Results
  are in `results/exp19_coactivity_rewire_seed{0,1,2}/`.

## 2026-10-07 — exp19 results: coactivity rewiring does not segregate branches either

**Branch metric.** same_branch_frac, mean over all 10 task pairs, last layer, init → final, mean ± std over 3 seeds:

| Arm (k=4) | Naive x | Replay x | Replay y |
|---|---|---|---|
| dense den_max | 0.990 → 0.971±0.010 | 0.990 → 0.947±0.021 | 0.990 → 0.953±0.010 |
| sparse fixed (16 contacts) | 0.992 → 0.977±0.007 | 0.992 → 0.956±0.004 | 0.992 → 0.949±0.001 |
| sparse coactivity rewire | 0.992 → 0.967±0.002 | 0.992 → 0.956±0.005 | 0.992 → 0.949±0.008 |

For comparison, the exp18 SET-rewire arm went 0.993 → 0.971±0.002 (naive x)
and 0.950±0.003 (replay x).

**Accuracy (final average) and forgetting (average excluding last task):**

| Arm | Naive acc | Naive forget | Replay acc | Replay forget |
|---|---|---|---|---|
| bdh_narrow | 0.198±0.001 | 0.996±0.001 | 0.840±0.041 | 0.190±0.052 |
| dense den_max | 0.198±0.001 | 0.998±0.001 | 0.864±0.015 | 0.163±0.014 |
| point fixed | 0.199±0.001 | 0.993±0.004 | 0.856±0.024 | 0.169±0.029 |
| point coactivity | 0.198±0.003 | 0.997±0.003 | 0.851±0.031 | 0.175±0.035 |
| den fixed | 0.199±0.001 | 0.996±0.003 | 0.859±0.013 | 0.168±0.015 |
| den coactivity | 0.200±0.002 | 0.995±0.003 | 0.858±0.010 | 0.169±0.011 |

**Verdict.** The null is the same as exp18. Coactivity-guided regrowth
leaves the dominant branch about 95–97% shared across tasks, the same as the
fixed mask and as SET. Naive is still readout collapse at about 0.20. Replay
shows no rewiring effect and no branch-vs-point effect beyond seed noise.
This contradicts the toy lab, where clustered regrowth was the ingredient
that raised segregation. The regrow-target rule is not the missing piece in
real BDH.

**Interpretation.** The difference between the toy and BDH is the learning
regime, not the regrowth rule. In the toy, only the winning branch of the
winning neuron updates, using a local rule, and the readout is a weight-free
label tally. In BDH, every active synapse gets a global gradient every step,
and the dense decoder and lm_head are retrained on every task. The next
tests should separate these two factors: a component swap or drift analysis
on the exp19 checkpoints, and a deeper toy with a learned readout.

**Not done.** proto_eval on the exp18/exp19 checkpoints (still on the
cluster). Checkpoints were not pulled.

## 2026-10-07 — Toy lab (Dendritic Memory Lab artifact v5): depth, learned readout, backprop

Browser toy (8×8 glyphs, 5 digit-pair tasks, 10 neurons × 4 branches), extended with stacked layers
(top-k soma passing between layers), a learned softmax readout (BDH lm_head analogue), a backprop
feature rule, and a readout-independent feature probe (nearest class mean on top-layer soma). Source in
the artifact (https://claude.ai/artifact/MibwiqjSoV2ZwSFVhtY9Bu); not in this repo. All 9 original presets
reproduce bit-identically under the new simulator. 3 seeds, mean±std:

| preset | acc | forgetting | learned-readout acc | probe |
|---|---|---|---|---|
| + clustered rewiring (1 layer, label-tally readout) | 0.492 | 0.035 | 0.253 | 0.883 |
| same features, learned softmax readout | 0.253±0.042 | 0.906±0.072 | 0.253 | 0.883 |
| two local layers (passK 1) | 0.403±0.100 | 0.229 | 0.181 | 0.536 |
| three local layers (passK 1) | 0.339±0.134 | 0.365 | 0.169 | 0.508 |
| backprop, two layers, learned readout (passK 10) | 0.281±0.017 | 0.892±0.016 | 0.281 | 0.544 |

- With features held fixed, the learned readout alone reproduces BDH-style collapse (forgetting 4% → 91%)
  while the probe stays at 88%: in the toy, forgetting lives in the readout.
- Backprop reaches ~0.99 on the task it just trained, then forgets 89%, and the probe drops to 0.54. This
  matches BDH naive (0.20 acc, ~99% forgetting).
- Stacking local competitive layers hurts on every metric. Caveat: upper layers reuse layer-1 constants.
  Without top-k passing, ~9.6/10 layer-2 units fire on every input.
- Failed variant: input jitter made every rule worse (probe ~0.19), so it was removed.
- Implication for BDH: run proto_eval on the exp18/19 checkpoints to test whether the readout-vs-feature
  split seen in the toy also holds there. Still not done.

### Follow-up: learned readout that reads branches (artifact v6)

Added `readFrom: "branch"`: the softmax reads all N×K top-layer branch activations, and under backprop the
gradient reaches each top branch directly instead of only through the max-soma winner. 3 seeds:

| preset | acc | forgetting | diag | probe | tally-on-branches acc |
|---|---|---|---|---|---|
| backprop 2 layers, readout on soma | 0.281±0.017 | 0.892±0.016 | 0.994 | 0.544±0.133 | 0.283 |
| backprop 2 layers, readout on branches | 0.297±0.083 | 0.878±0.104 | 1.000 | 0.747±0.089 | 0.317 |
| backprop 1 layer, soma / branches | 0.392 / 0.425 | 0.760 / 0.719 | 1.0 | 0.897 / 0.922 | – |
| local features, learned readout on soma | 0.253±0.042 | 0.906±0.072 | – | 0.883 | – |
| local features, learned readout on branches | 0.394±0.017 | 0.743±0.042 | – | 0.883 | – |

- Branch identity improves backprop features (probe +0.20) but not forgetting (within seed noise), and layer-1
  same-branch fraction does not drop meaningfully (0.64 → 0.59 ± 0.10).
- The label-tally readout's 4% forgetting comes from never overwriting old weights, not from seeing branches.
- Candidate BDH test (not run): lm_head on pre-soma branch activations. The toy predicts a better proto_eval
  score but no change in naive forgetting.

## 2026-10-07 — Step 2 (toy): gated backprop + frozen-prototype readout

Toy simulator (artifact source, not in repo) gained `gradTopK` (only the k most active branches per layer get
gradient), `taskGate` (each task writes only to its own random 1/5 of branches; task id at train time only),
and a frozen-prototype readout (class means stored at the end of the class's own task, never refreshed —
same protocol as `proto_eval.py` "frozen"). 3 seeds (1,2,3), mean±std:

| arm | learned acc | learned fgt | diag | frozen-proto acc | frozen-proto fgt |
|---|---|---|---|---|---|
| local full rule (1 layer) | 0.253±0.042 | 0.906 | 0.978 | 0.736±0.069 | 0.306±0.087 |
| backprop 1 layer, ungated | 0.392±0.068 | 0.760 | 1.000 | 0.728±0.068 | 0.333±0.073 |
| backprop 1 layer, top-k k=1/2/4/8 | 0.21–0.38 | 0.77–0.94 | 0.96–1.0 | 0.63–0.73 | 0.32–0.38 |
| backprop 1 layer, task gate | 0.211±0.029 | 0.962 | 0.981 | 0.703±0.039 | 0.240±0.063 |
| backprop 1 layer, task gate + consolidation | 0.189±0.013 | 0.976 | 0.969 | 0.736±0.035 | 0.191±0.067 |
| backprop 2 layers, ungated | 0.281±0.017 | 0.892 | 0.994 | 0.525±0.130 | 0.542±0.171 |
| backprop 2 layers, top-k k=1..4 | 0.12–0.13 | 0.60–0.66 | 0.60–0.66 | 0.33–0.38 | 0.18–0.21 |
| backprop 2 layers, task gate | 0.142±0.044 | 0.715 | 0.714 | 0.394±0.079 | 0.222±0.084 |

- Activity top-k gating does not help: the same strongly-responding branches win for every task. In 2 layers
  it lowers "forgetting" only by starving learning (diag 0.60).
- Task-partitioned gating helps the features a little in 1 layer (frozen-proto forgetting 33% → 24%, 19% with
  consolidation, at the same accuracy) but the shared learned head still collapses (96%).
- The big lever in the toy is the readout: frozen prototypes on ungated 1-layer backprop features give 0.73,
  the same as the local rule (0.74). In the toy, backprop features are fine; the head is the problem.
- This does NOT transfer to BDH: see proto results below (frozen ≈ 0.20 under naive).

## 2026-10-07 — Step 1: proto_eval on exp18/exp19 checkpoints (jobs 4384314, 4384320)

Eval only (`slurm/proto_exp1819.sbatch`, `slurm/proto_exp18_rerun.sbatch`), 3 seeds, results in
`bdh_cl/results/proto_exp18_19/`. Job 4384314's smoke step passed. Its exp18 sparse arms crashed: those
checkpoints predate the `coactivity` buffer. Fix: `proto_eval.load` zero-fills only a missing `coactivity`
buffer, which is a training-time trace never read in forward; any other missing key still raises
(`tests/test_proto_eval.py`, 2 passed). Then exp18 was rerun as job 4384320.

| arm | naive frozen | replay frozen | replay refreshed (oracle) | replay drift |
|---|---|---|---|---|
| narrow (exp18 / exp19) | 0.198 / 0.197 | 0.570±0.097 / 0.588±0.101 | 0.882 / 0.892 | +0.31 / +0.30 |
| den_max (exp18 / exp19) | 0.198 / 0.197 | 0.602±0.048 / 0.605±0.046 | 0.911 / 0.909 | +0.31 / +0.30 |
| k=1 sparse fixed (exp18 / exp19) | 0.200 / 0.200 | 0.597±0.085 / 0.588±0.090 | 0.895 / 0.899 | +0.30 / +0.31 |
| k=1 SET rewire (exp18) | 0.199 | 0.528±0.096 | 0.887 | +0.36 |
| k=1 coactivity rewire (exp19) | 0.198 | 0.548±0.097 | 0.883 | +0.34 |
| k=4 sparse fixed (exp18 / exp19) | 0.198 / 0.197 | 0.502±0.045 / 0.499±0.032 | 0.888 / 0.895 | +0.39 / +0.40 |
| k=4 SET rewire (exp18) | 0.198 | 0.526±0.032 | 0.893 | +0.37 |
| k=4 coactivity rewire (exp19) | 0.197 | 0.461±0.032 | 0.902 | +0.44 |

- Naive: every arm reaches 0.197–0.200 with frozen prototypes, the same as the collapsed head. On BDH the features
  are lost, not just the readout. This confirms Phase 5 and contradicts the toy, where 1-layer backprop features
  survive (frozen-proto 0.73).
- Replay: k=4 sparse dendrites drift more than dense den_max (+0.39–0.44 vs +0.30), and coactivity rewiring has the
  worst frozen score (0.46). Sparse and rewired branches make features less stable, not more.
- Shared-machinery sanity check: arms present in both exps (narrow, den_max, sparse_fixed) agree within noise.

## 2026-10-07 — Toy: tiredness-guided binary population codes (user's idea)

Standalone toy (scratchpad `deep/tired.js`, same glyph data and 5 digit-pair tasks as the lab). Setup:
N neurons × 4 branches (16 contacts each, max soma), binary top-k output, backprop with a sigmoid
straight-through surrogate through a CE head. Tiredness t = f/(f+20), where f is the neuron's firing count.
Arms:
- B: λ·Σt·a added to the loss
- D: lr × (1−t)
- C: both B and D
- E: top-k on (drive − 0.5·t), plus D
- F: fixed random weights
Readouts store the combination at the end of each class's own task: frozen prototypes, and "pattern"
(neurons active on ≥50% of probe examples, scored by the fraction of the pattern that is on). 3 seeds:

| N / k | arm | proto acc | proto fgt | diag | pattern acc | head acc |
|---|---|---|---|---|---|---|
| 200/10 | A no tiredness | 0.942±0.042 | 0.031 | – | 0.836±0.038 | 0.539 |
| 200/10 | B penalty λ=0.5 | 0.653±0.046 | 0.267 | – | 0.461±0.075 | 0.319 |
| 200/10 | C penalty+plastic | 0.869±0.072 | 0.059 | – | 0.703±0.034 | 0.461 |
| 200/10 | D plastic only | 0.928±0.029 | 0.024 | – | 0.800±0.036 | 0.500 |
| 200/10 | E select+plastic | 0.942±0.014 | 0.017 | – | 0.825±0.036 | 0.592 |
| 200/10 | F fixed random | 0.933±0.030 | 0.028 | – | 0.767±0.029 | 0.483 |
| 40/4 | A | 0.775±0.036 | 0.156 | 0.889 | 0.714 | 0.386 |
| 40/4 | C λ=0.1 | 0.728±0.039 | 0.118 | 0.822 | 0.664 | 0.333 |
| 40/4 | E | 0.761±0.046 | 0.087 | 0.831 | 0.653 | 0.358 |
| 40/4 | F | 0.706±0.024 | 0.111 | 0.794 | 0.594 | 0.300 |
| 20/2 | A | 0.639±0.047 | 0.281 | 0.864 | 0.578 | 0.253 |
| 20/2 | C λ=0.1 | 0.606±0.025 | 0.142 | 0.719 | 0.553 | 0.250 |
| 20/2 | D | 0.617±0.025 | 0.118 | 0.711 | 0.564 | 0.256 |
| 20/2 | F | 0.594±0.060 | 0.167 | 0.728 | 0.386 | 0.219 |

- Tiredness as a loss term (B) separates the tasks' neurons (overlap 0.17 → 0.01 at λ=0.5) but wrecks
  accuracy, because switching tired neurons off rewrites their old codes. At λ=0.1 it does nothing.
- Tiredness-scaled plasticity (C/D/E) halves forgetting at tight capacity (N=20: 28% → 12–14%) but lowers
  diag by the same amount (0.86 → 0.71). Final accuracy does not improve: a stability–plasticity trade, not a gain.
- What works without any tiredness: a wide binary top-k population plus a stored-combination readout. At N=200
  plain backprop gets 0.94 with 3% forgetting, and even fixed random weights get 0.93. Population size is the
  lever (N=20 → 40 → 200: 0.64 → 0.78 → 0.94).
- Caveats: one layer, easy glyph task; the BDH failure is deep-feature drift, which this toy does not have.

### 2026-10-07 — Sanity baseline: the toy task is trivial

Nearest class mean on **raw pixels**: cosine similarity to the class means of the same
12 probe examples per class, argmax over all 10 classes, scored on the same noisy test
sampler. Accuracy is 1.000, 0.992, 0.992 (seeds 1–3), so **0.995 ± 0.005**.
This beats every tiredness arm (best 0.942), so the toy's 0.94 is not evidence of continual learning.
It only shows that prototypes stored at task end don't drift when features are fixed or barely move.
Any claim about challenging continual learning needs a benchmark where raw or random features + NCM are weak.

### 2026-10-07 — Training-free class-incremental floor (job 4385548)

Script `bdh_cl/scripts/ncm_baselines.py` (tests in `test_ncm_baselines.py`, 4 passed locally and on the cluster). Setup:
- Nearest class mean (cosine), argmax over all seen classes, no task id.
- Class means come from all training images of that class.
- Features are fixed, so stored prototypes never drift and there is zero forgetting by construction.
- Random maps (pixels centred per image): `rand_kwta` is a Gaussian projection followed by binary top-5%; `rand_dend` uses 4 branches × 16 random contacts per neuron, max over branches, then binary top-5%.
- 3 seeds (seed = random weights; raw pixels are deterministic).
- The CIFAR-100 smoke run (2,000 images) passed first.

Final accuracy = average over tasks after the last task. Avg inc = average incremental accuracy.

| features | Split MNIST 5×2: final | Split MNIST 5×2: avg inc | Split CIFAR-100 10×10: final | Split CIFAR-100 10×10: avg inc |
|---|---|---|---|---|
| raw pixels | 0.817 | 0.897 | 0.118 | 0.172 |
| rand kWTA N=2000 | 0.830±0.005 | 0.908±0.004 | 0.134±0.001 | 0.198±0.002 |
| rand dend N=2000 | 0.821±0.004 | 0.900±0.004 | 0.146±0.002 | 0.213±0.002 |
| rand kWTA N=10000 | 0.845±0.002 | 0.915±0.002 | 0.139±0.001 | 0.203±0.001 |
| rand dend N=10000 | 0.848±0.002 | 0.917±0.001 | **0.151±0.001** | 0.218±0.001 |

Reading:
- Split MNIST is weak evidence: memorising class means with no learning already gets about 0.85 final accuracy, and that number never forgets.
- On CIFAR-100, random features plus NCM manage only about 0.15. The task becomes hard as soon as features have to be learned, so it is the right benchmark.
- Random dendritic branches beat a plain random projection on CIFAR (+1.2pt), which is small but consistent across 3 seeds.
- Target for a learned population code: clearly above 0.15 final on CIFAR-100 10×10 with no exemplars. It also needs to be compared with fine-tuning, LwF and EWC run in the same harness. We have no numbers for those yet, so none are quoted here.

### 2026-10-07 — Exp20: population code vs CL baselines, Split CIFAR-100 10×10 (tuning, validation split, seed 0)

Script `bdh_cl/scripts/pop_cl_cifar.py`; tests `test_pop_cl_cifar.py` (11 passed).

**Test-writing found a real bug.** z-scoring the scores and the k-th-score threshold leaked gradient across neurons, so neurons whose plasticity had been scaled to 0 were still updated. Fixed by detaching the per-example mean, std and threshold.

**First smoke failed.** Job 4386551 hit a native abort in CPU conv backward on a GPU node; it did not reproduce on the login node. The end-to-end tests now run on CUDA when it is available. Smoke job 4387458: all 7 arms OK.

**Tuning job 4387491.**
- Setup: reduced ResNet-18 (nf=20), SGD lr 0.05, momentum 0.9, wd 5e-4, cosine schedule per task, 20 epochs per task, batch 128, crop+flip augmentation, no exemplars.
- Evaluation is over all seen classes.
- `ncm` = prototypes frozen at the end of each class's own task.
- `oracle` = prototypes recomputed with the final model (not a method).
- Each run takes about 1 minute.

| run | head | ncm (frozen) | oracle | drift | NCM forgetting |
|---|---|---|---|---|---|
| finetune | 0.078 | 0.076 | 0.143 | 0.066 | 0.81 |
| finetune_local | 0.098 | 0.100 | 0.193 | 0.093 | 0.76 |
| lwf λ=1 | 0.086 | 0.167 | 0.339 | 0.172 | 0.62 |
| **lwf λ=3** | 0.079 | **0.184** | 0.333 | 0.149 | 0.50 |
| lwf λ=10 | 0.075 | 0.151 | 0.263 | 0.111 | 0.41 |
| ewc λ=10 | 0.077 | 0.152 | 0.310 | 0.158 | 0.67 |
| ewc λ=100, 1e3, 1e4 | 0.010 | 0.010 | — | — | — |
| pop k=50 | 0.093 | 0.093 | 0.160 | 0.067 | 0.73 |
| pop k=100 | 0.096 | 0.095 | 0.206 | 0.111 | 0.74 |
| pop k=200 | 0.100 | 0.090 | 0.242 | 0.152 | 0.74 |
| pop_tired τ=0.25 | 0.113 | 0.118 | 0.229 | 0.111 | 0.55 |
| pop_tired τ=1 | 0.107 | 0.110 | 0.223 | 0.113 | 0.64 |
| pop_tired τ=4 | 0.102 | 0.108 | 0.220 | 0.112 | 0.68 |
| joint (upper bound) | 0.639 | — | 0.610 | — | — |

**EWC failure.** EWC with λ≥100 diverged: the loss went NaN at task 2. That is an instability in my implementation (penalty × lr 0.05 with momentum), not a property of EWC. It needs gradient clipping or a lower lr before EWC is a fair baseline.

**Reading (one seed, validation split; to be confirmed with 3 seeds):**
- Training-free floor: random dendritic features + NCM = 0.151 on the test split (job 4385548).
- **Nothing here beats that floor convincingly.** LwF+NCM at 0.184 is the only arm above it, by a small margin.
- **The population code as built does not help.** pop with frozen NCM (0.09) is no better than finetune_local (0.10), and both sit below random features. The binary top-k layer does not protect the backbone, and drift happens in the backbone.
- **Tiredness-scaled plasticity is the one population ingredient that moves the number.** It gives +1.5–2.3pt over pop and cuts NCM forgetting from 0.74 to 0.55 at τ=0.25. The effect is consistent in direction but small, and still below the floor.
- **The oracle gap is large in every arm.** Old classes stay decodable (0.2–0.34) but their stored prototypes go stale. As in BDH Step 1, the failure is drift, not erasure.

### 2026-10-07 — Exp20b: no convolution (job 4387567, validation split, seed 0)

Population layers sit directly on pixels (`--backbone none`), so every learned weight is inside the branch/top-k/tiredness mechanism. The control is a dense ReLU layer of the same width (`--backbone mlp`). Same optimiser and epochs as Exp20. Tests: 17 passed.

| run | head | ncm (frozen) | oracle | NCM forgetting | task-0 acc |
|---|---|---|---|---|---|
| joint, dense MLP (ceiling) | 0.328 | — | — | — | — |
| joint, pop 1 layer (ceiling) | 0.264 | — | — | — | — |
| joint, pop 2 layers (ceiling) | 0.252 | — | — | — | — |
| finetune_local, MLP | 0.088 | 0.073 | 0.169 | 0.48 | 0.448 |
| lwf λ=3, MLP | NaN from task 2 | | | | |
| pop L1 | 0.101 | 0.095 | 0.167 | 0.43 | 0.464 |
| pop L2 | 0.091 | 0.095 | 0.180 | 0.50 | 0.504 |
| **pop_tired L1 τ=0.25** | 0.124 | **0.141** | 0.167 | **0.23** | 0.464 |
| pop_tired L1 τ=1 | 0.116 | 0.121 | 0.174 | 0.33 | 0.464 |
| pop_tired L2 τ=0.25 | 0.116 | 0.131 | 0.159 | 0.24 | 0.504 |

**Reading:**
- **Once tiredness controls every learned weight, it clearly works as a stabiliser.** NCM forgetting halves (0.43 → 0.23), and frozen-prototype accuracy rises from 0.095 to 0.141, almost reaching the oracle (0.167). Drift nearly disappears.
- **But it ends at roughly the level of untrained random features** (0.151 on the test split; not yet measured on validation). By the end, 86% of neurons are more than 90% tired, so the network has effectively frozen.
- **The real limit is feature quality, not forgetting.** Even the oracle (fresh prototypes, final model) is only 0.17, against a joint ceiling of 0.26–0.33. Features learned one 10-class task at a time are narrow, and freezing them preserves narrow features.
- **The no-conv ceiling is low.** The best possible here is 0.26–0.33, so this setting cannot show strong continual learning however good the mechanism is.
- **LwF on the MLP went NaN at task 2.** This is the same instability as EWC in Exp20: lr 0.05 with momentum and an added loss term. It needs gradient clipping before LwF/EWC are fair baselines.

### 2026-10-07 — Exp20c: wide population + tiredness, no conv (job 4387785, validation split, seed 0)

The untrained floors use the same population layer, preprocessing and prototype readout as the trained arms (`--epochs 0`), so they are matched baselines on the same validation split.

| run | ncm final | ncm avg inc | oracle | NCM forgetting | neurons >0.9 tired |
|---|---|---|---|---|---|
| untrained N=2000 k=100 | 0.129 | 0.192 | 0.129 | 0.09 | — |
| untrained N=10000 k=200 | 0.130 | 0.199 | 0.130 | 0.09 | — |
| pop N=8000 k=160 (no tiredness) | 0.083 | 0.177 | 0.161 | 0.44 | — |
| tired N=2000 k=40 τ=0.25 | 0.131 | 0.220 | 0.139 | 0.15 | 0.34 |
| tired N=2000 k=100 τ=0.25 (Exp20b) | 0.141 | — | 0.167 | 0.23 | 0.86 |
| tired N=8000 k=160 τ=0.25 | 0.140 | 0.230 | 0.163 | 0.20 | 0.68 |
| tired N=8000 k=160 τ=1 | 0.120 | 0.219 | 0.173 | 0.31 | 0.38 |
| tired N=10000 k=200 τ=0.25 | 0.140 | 0.232 | 0.159 | 0.21 | 0.74 |
| tired N=10000 k=100 τ=0.25 | 0.133 | 0.227 | 0.153 | 0.17 | 0.35 |

**Reading (one seed):**
- **Width does not help.** Final accuracy is 0.140–0.141 at N=2000, 8000 and 10000.
- **Best trained arm vs matched random floor:** final accuracy is +1.1pt (0.141 vs 0.130). That is about 2 standard errors on 5,000 validation images. Average incremental accuracy is +3–4pt (0.23 vs 0.19–0.20).
- **Without tiredness, training is worse than not training at all** (0.083 vs 0.130). Tiredness is what turns learning from harmful into roughly neutral.
- **Verdict:** the population + tiredness mechanism on pixels sits at the random-feature floor. Fresh capacity does not fix it. Learned features from 10-class tasks are only slightly better than random ones and are no longer improving at the end. This does not support a claim of strong continual learning on CIFAR-100. Direction parked unless something new changes the feature-learning side.

### 2026-10-07 — Exp20d: population + tiredness on FROZEN ImageNet ResNet-18 features (job 4387886, validation split, seed 0)

**Features.** `extract_feats.py` runs torchvision ResNet-18 (IMAGENET1K_V1) on CIFAR images upsampled to 224 and keeps the 512-d avgpool output. Shapes: train 45000, valid 5000, test 10000. Weights are at `$WS/data_cache/torch_hub`; features at `$WS/data_cache/cifar100_r18_feats.pt`.

**Tests:** 24 passed. The extract smoke run (64 images) was OK.

**Caveat:** this is the pretrained continual-learning regime. ImageNet overlaps CIFAR-100's concepts.

**Training:** same as before (SGD lr 0.05, momentum 0.9, 20 epochs per task). Not retuned for features.

| run | head | ncm (frozen) | ncm avg inc | oracle | NCM forgetting |
|---|---|---|---|---|---|
| **NCM directly on frozen features (no learning)** | — | **0.543** | 0.644 | 0.543 | 0.12 |
| joint linear probe (ceiling) | 0.684 | — | — | — | — |
| joint pop N=2000 (ceiling) | 0.670 | — | — | — | — |
| untrained pop N=2000 k=100 | — | 0.499 | 0.599 | 0.499 | 0.12 |
| untrained pop N=8000 k=160 | — | 0.524 | 0.624 | 0.524 | 0.12 |
| finetune_local MLP | 0.225 | 0.166 | 0.335 | 0.526 | 0.81 |
| lwf λ=3 MLP, clip 1.0 | 0.089 | 0.085 | 0.292 | 0.482 | 0.88 |
| pop N=2000 | 0.249 | 0.267 | 0.473 | 0.512 | 0.68 |
| pop_tired N=2000 τ=0.25 | 0.319 | 0.345 | 0.536 | 0.533 | 0.50 |
| pop_tired N=2000 τ=1 | 0.327 | 0.357 | 0.521 | 0.539 | 0.51 |
| **pop_tired N=8000 τ=0.25** | 0.352 | **0.419** | 0.598 | 0.547 | 0.38 |

**Reading:**
- **Every learned arm is below "store class means of the frozen features" (0.543).** Learning on top of good features destroys information faster than it adds any.
- **Tiredness again helps relative to no tiredness.** At N=2000 it lifts frozen NCM from 0.267 to 0.345–0.357. At N=8000 it reaches 0.419 and cuts forgetting from 0.68 to 0.38.
- **Here, unlike on pixels, width helps.** At τ=0.25, N=8000 beats N=2000 (0.419 vs 0.345).
- **Everything is still more than 12pt below doing nothing.**
- **The oracle is roughly equal across arms (0.51–0.55).** The information is still there; frozen prototypes go stale.
- **LwF with clipping no longer NaNs, but it is badly hurt at lr 0.05 on features.** It is not a tuned baseline.
- **Implication:** in the pretrained regime the competitor is analytic, gradient-free accumulation (NCM / streaming LDA / ridge), not gradient training. The room to improve is the 14pt gap between NCM (0.543) and the joint ceiling (0.684).

## Exp21 — untrained population code + analytic readouts on frozen ResNet-18 features (2026-10-07, job 4387921)

Question: Exp20d showed no gradient-trained arm beats NCM on frozen features. Is the
binary dendritic population a better *fixed expansion* for gradient-free, statistics-
accumulating readouts (NCM / streaming LDA / ridge = RanPAC-style) than raw features or a
ReLU random projection? Every readout is the exact joint solution on tasks 0..t (tests
check streaming == batch), so forgetting comes only from new classes competing in argmax.
Script `bdh_cl/scripts/analytic_readout.py`, tests `test_analytic_readout.py` (4 pass).
Shrinkage/lambda chosen on valid (final acc), test reported, 3 seeds (raw is deterministic).
First submission 4387916 failed a test: PopLayer's straight-through output is not exactly
0/1 in float; the map now thresholds at 0.5. Total compute ~3 min.

Split CIFAR-100 10x10, class-incremental, **test**, final avg acc (avg incremental in parens):

| map | dim | NCM | ridge | streaming LDA |
|---|---|---|---|---|
| raw ResNet-18 | 512 | 0.540 (0.643) | 0.621 (0.729) | 0.639 (0.736) |
| ReLU proj | 2000 | 0.535 | 0.653±0.002 | 0.649±0.001 |
| ReLU proj | 8000 | 0.541 | **0.692±0.003** (0.746) | 0.675±0.004 |
| pop N2000 k100 K4 | 2000 | 0.501 | 0.538±0.001 | 0.559±0.001 |
| pop N8000 k160 K4 | 8000 | 0.524 | 0.584±0.000 | 0.594±0.003 |
| pop N8000 k160 K1 | 8000 | 0.518 | 0.577±0.003 | 0.589±0.001 |
| pop N8000 k400 K4 | 8000 | 0.539 | 0.627±0.002 | 0.625±0.001 |

Reference: joint linear probe 0.684 (valid, Exp20d); best trained CL arm 0.419 (pop_tired N8000, valid).

Findings:
- Analytic readouts beat every trained CL arm by a wide margin. ReLU-8000 + ridge (= RanPAC
  recipe) reaches 0.692, at the joint linear-probe level, with zero gradient training.
- The binary top-k population code is a *worse* expansion than ReLU at matched width:
  -10.8pt (k160) / -6.5pt (k400) vs ReLU-8000 ridge, and below raw-feature LDA (0.639).
  Binarising throws away magnitude; denser codes (k400 > k160 > k100) monotonically help,
  i.e. the sparsity is the cost, not a benefit.
- Branches help only marginally (K4 vs K1: +0.5–0.7pt).
- Caveats: raw ridge picked the grid edge (1e-4) and ReLU-8000 seed1 also 1e-4 — lower
  lambdas untested, so those two may be slightly underestimated. Seeds vary only the random
  map; features/splits fixed.

## Exp22 — continual RL on MinAtar (2026-10-07/08)

Setup (`bdh_cl/scripts/pop_cl_rl.py`, tests `test_pop_cl_rl.py`, 16 pass): one DQN agent
(conv16 -> hidden 512 -> 6 actions) plays breakout -> space_invaders -> freeway -> asterix
-> seaquest, twice (10 phases x 200k env steps). Replay emptied and epsilon re-annealed at
each switch; weights, Adam state, target net carry over. Obs padded to 10 channels, so
channel meaning changes per game. Plasticity = train return (last 25% of phase) in cycle 2
vs cycle 1 and vs a fresh agent on that game alone; forgetting = greedy eval on all games
after each phase. Arms: dqn, dqn_redo (ReDo, thresh 0.1, every 1000 updates), pop
(PopLayer N512 K4 k32), pop_tired (tau = 1 x expected per-phase firings), pop_reset,
pop_tired_reset. MinAtar 1.0.15 pip-installed in the cluster venv (seed via env.seed()).

Smoke (job 4387992, 2 games x 10k steps): all arms run, ~1000 env steps/s -> ~40 min/run.
- dqn: 39-44% of hidden ReLU units dormant already after 10k steps; ReDo holds ~10%.
- pop: 70-88% of neurons never fire in a window -- top-32 winners concentrate on a small
  fixed set (MinAtar frames are very similar).
- **Design flaw found:** pop_reset re-draws ~2200 neurons per 10k-step phase and dormancy
  stays ~0.85 -- freshly drawn neurons cannot beat the trained winners into the top-k, so
  "reset silent neurons" just churns most of the layer. Reset arms held back; they need a
  recruitment mechanism that lets new neurons win (e.g. usage-based excitability /
  homeostatic boosting) before they are a fair test.

Main array 4387996: {dqn, dqn_redo, pop, pop_tired} x 3 seeds + fresh-agent dqn per game x 3 seeds.

### Exp22 results (array 4387996, 3 seeds each, all 15 tasks finished; analysis `analyze_exp22.py`)

Train return in last 25% of each 200k-step phase, mean±std over seeds. fresh = dqn on that
game alone: breakout 6.42, space_invaders 24.36, freeway 24.28, asterix 1.27, seaquest 0.40.
plast = mean over games of (cycle-2 return / fresh).

| arm | breakout c1→c2 | space_inv c1→c2 | freeway c1→c2 | plast (c2/fresh) | c2/c1 | dormant end |
|---|---|---|---|---|---|---|
| dqn | 7.10→5.80 | 17.7→11.8 | 12.6→14.4 | **0.71±0.02** | 0.89±0.08 | 0.96 |
| dqn_redo | 6.10→5.60 | 17.4→17.9 | 21.4→22.0 | **1.00±0.02** | 1.01±0.04 | 0.52 |
| pop | 2.35→1.91 | 4.6→4.5 | 4.4→4.2 | 0.33±0.06 | 0.98±0.19 | 0.91 |
| pop_tired | 2.57→2.26 | 6.6→5.4 | 4.7→2.7 | 0.34±0.06 | 0.86±0.14 | 0.91 |

(asterix/seaquest barely learned by anyone in 200k steps -- fresh 1.27/0.40 -- so their ratios are noise.)

Findings:
- Plasticity loss is real here: plain DQN learns games ~30% worse than a fresh agent by
  cycle 2 (already in cycle 1: freeway 12.6 vs fresh 24.3), with 96% of ReLU units dormant.
  ReDo fully restores it (1.00), reproducing Sokar et al. 2023.
- The binary top-k population layer is a much weaker Q-learner from the very first game:
  breakout phase 0 (effectively a fresh agent) 2.35 vs 7.10 for DQN; space_invaders 4.6 vs 17.7.
  ~90% of its neurons never fire in a phase. Its flat c2/c1 (0.98) is not preserved plasticity
  -- it never learned much. Tiredness changes nothing (0.34). Population arms cannot be judged
  on plasticity until the layer itself can learn Q-values.
- Forgetting is near-total for every arm (final greedy eval on earlier games ~0-2 vs fresh 6-24),
  as expected with one shared network and no replay of old games.
- Reset arms (pop_reset, pop_tired_reset) not run -- see design flaw above; moot now given pop's
  baseline weakness.

## Exp23 — does homeostatic excitability let the binary population layer learn? (2026-10-08, array 4392250)

Hypothesis: the pop layer learns Q-values badly (Exp22) because ~90% of neurons never win
the top-k and so never get gradient. Fix tried: `HomeoPop` in `pop_cl_rl.py` -- per-neuron
offset e added to the z-score before top-k, e += eta*(k/N - r) with r an EMA firing rate
(eta 0.05, alpha 1e-3); `_up` variant clamps e >= 0 (boost idle, never suppress busy).
Tests: 24 pass. Fresh agent, ONE game, 200k steps, 3 seeds; dqn reference = Exp22 fresh runs.
All 18 runs finished.

| game | arm | train return | dormant (last window) |
|---|---|---|---|
| breakout | dqn | 6.42±0.57 | 0.11 |
| breakout | pop | 2.62±0.18 | 0.89 |
| breakout | pop_homeo | 2.70±0.07 | 0.00 |
| breakout | pop_homeo_up | 2.67±0.08 | 0.00 |
| space_invaders | dqn | 24.36±0.92 | 0.62 |
| space_invaders | pop | 6.07±0.13 | 0.91 |
| space_invaders | pop_homeo | 5.92±0.44 | 0.00 |
| space_invaders | pop_homeo_up | 6.40±0.12 | 0.00 |

**Negative result.** Homeostasis does exactly what it should mechanically -- silent neurons
go from ~90% to 0%, every neuron takes part -- and performance does not move (within noise
of plain pop on both games, still 2.4x / 4x below DQN). So silent neurons are NOT why the
binary layer learns poorly. Remaining suspects: binary output discarding magnitude (Exp21:
-6.5 to -11pt vs ReLU at matched width, denser k better), the straight-through surrogate
gradient, and k=32/512 being too sparse for value regression. Not yet separated.

## Exp24 — which part of the binary top-k layer blocks Q-learning? (2026-10-08, array 4392855)

`GradedPop` in `pop_cl_rl.py`: same top-k winners, winner outputs z - theta (> 0) instead of 1.
`graded` = true gradient through winners; `gradedst` = graded forward, PopLayer's sigmoid
surrogate backward (isolates forward magnitude). Tests: 30 pass. Fresh agent, one game,
200k steps, 3 seeds, all 24 runs finished. Train return (last 25%), mean±std:

| variant | active | breakout | space_invaders | dormant (bo / si) |
|---|---|---|---|---|
| dqn (ReLU 512) | dense | 6.42±0.57 | 24.36±0.92 | 0.11 / 0.62 |
| binary k32 | 6% | 2.62±0.18 | 6.07±0.13 | 0.89 / 0.91 |
| binary k128 | 25% | 5.65±0.37 | 13.76±1.47 | 0.26 / 0.47 |
| graded k32, surrogate grad | 6% | 2.70±0.14 | 17.58±0.46 | 0.90 / 0.90 |
| graded k32, true grad | 6% | 2.73±0.20 | 16.02±0.92 | 0.93 / 0.90 |
| graded k128, true grad | 25% | 6.27±0.67 | 22.28±1.29 | 0.13 / 0.35 |

Findings:
- Sparsity (k) is the main bottleneck: k32 -> k128 roughly doubles both games while binary.
- Magnitude matters too, game-dependent: no effect on breakout at k32, x2.7 on space_invaders.
- Surrogate vs true gradient: no meaningful difference -> the surrogate was not the problem.
- Graded + k128 reaches DQN on breakout (6.27 vs 6.42) and ~91% of it on space_invaders
  (22.3 vs 24.4, about 1.5 std apart at n=3). Greedy eval agrees (10.9 vs 6.7; 24.9 vs 24.9).
- Caveats: 2 games, 3 seeds, single-task only. 25% active is far less sparse than the
  original 6%, so task separation (what tiredness relies on) is weaker; untested.
