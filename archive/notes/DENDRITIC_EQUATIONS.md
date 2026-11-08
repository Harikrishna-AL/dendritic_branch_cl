# How the BDH equations change with dendrites

Reference: upstream `bdh.py` (`2b0d7a4`), implemented in
`bdh_cl/models/neuron_units.py::DendriticNeuronUnit`.

Shapes: `B` batch, `T` context, `D = n_embd`, `nh = n_head`,
`N = mult·D/nh` per-head latent width, `n = nh·N` total latent neurons,
`k` dendritic branches per neuron.

---

## Baseline BDH, one layer

All three weight matrices are **shared across layers** — the loop reuses them,
so `n_layer` does not change the parameter count.

**1. Query-path neurons**

$$x_{\text{sparse}} = \mathrm{ReLU}(x\,E), \qquad E \in \mathbb{R}^{nh \times D \times N}$$

`x: (B,1,T,D) → x_sparse: (B,nh,T,N)`

**2. Linear attention** — no softmax, no scaling, current token excluded

$$S = \mathrm{tril}_{-1}\!\big( \mathrm{RoPE}(x_{\text{sparse}})\, \mathrm{RoPE}(x_{\text{sparse}})^{\top} \big), \qquad y_{KV} = \mathrm{LN}(S\,x)$$

`S: (B,nh,T,T)`, `y_KV: (B,nh,T,D)`. Note `Q = K = x_sparse` and `V = x`.

**3. Value-path neurons**

$$y_{\text{sparse}} = \mathrm{ReLU}(y_{KV}\,E_v), \qquad E_v \in \mathbb{R}^{nh \times D \times N}$$

**4. Multiplicative gate** (this is BDH's "synapse")

$$xy = x_{\text{sparse}} \odot y_{\text{sparse}}$$

**5. Read-out and residual**

$$y_{\text{MLP}} = \mathrm{reshape}(xy)\,W_{\text{dec}}, \quad W_{\text{dec}} \in \mathbb{R}^{n \times D}, \qquad x \leftarrow \mathrm{LN}\big(x + \mathrm{LN}(y_{\text{MLP}})\big)$$

---

## Dendritic BDH

**Only steps 1 and 3 change. Steps 2, 4 and 5 are untouched** — attention still
consumes `x_sparse`, the gate still multiplies the two streams, the read-out
still sees one value per neuron. That is what makes this a drop-in swap and
why every existing probe and metric still applies.

**1′. Query-path neurons, with branches**

$$b = x\,E^{\text{den}}, \qquad E^{\text{den}} \in \mathbb{R}^{nh \times D \times k \times N}$$

$$a_c = \mathrm{ReLU}\big(b_c - \theta_c\big), \qquad \theta \in \mathbb{R}^{nh \times k \times N} \ \text{(optional)}$$

$$x_{\text{sparse}} = \underset{c=1..k}{\mathrm{soma}}\ a_c$$

`b, a: (B,nh,T,k,N) → x_sparse: (B,nh,T,N)` — same shape out as before.

**3′. Value path**: identical, with its own `E_v^den`, `θ_v`.

### Soma variants

| soma | formula | gradient reaching branch *c* |
|---|---|---|
| `sum` | $\sum_c a_c$ | every branch with $a_c>0$ |
| `max` | $\max_c a_c$ | **only the argmax branch** |
| `gated` | $\sum_c \sigma_c a_c$, $\sigma_c=\mathrm{softmax}_c(\beta_j a_c)$ | weighted by $\sigma_c$ |

### `gated` is the general form

$$\sigma_{j,c} = \mathrm{softmax}_c\big(\beta_j\, a_{c,j}\big), \qquad \beta_j = \mathrm{softplus}(\rho_j) > 0$$

with **one learnable $\beta$ per neuron**:

* $\beta_j \to 0$: $\sigma$ uniform, output is the mean over branches — the
  neuron behaves as a single ordinary unit, i.e. the plain BDH neuron.
* $\beta_j \to \infty$: $\sigma$ one-hot at the argmax, output $\to \max_c a_c$ —
  hard routing.

So the network chooses, per neuron, how much branch structure to use versus
behaving like an ordinary neuron. Verified numerically in
`test_gated_soma_interpolates_between_plain_neuron_and_hard_routing`.

Cost: $\beta \in \mathbb{R}^{nh \times N}$ per path, i.e. $2n$ parameters
(~0.1% of the backbone) — cheaper than the branch weights by a factor of
$kD$.

> **The gate must depend on the input.** An earlier version of this used a
> static per-neuron gate, a bare $(nh, k, N)$ parameter. That is useless: with
> fixed $\sigma_c > 0$,
> $\sigma_c\,\mathrm{ReLU}(b_c) = \mathrm{ReLU}\big(x\,(\sigma_c w_c)\big)$,
> so the gate folds into the branch weights and the unit collapses to `sum`
> with rescaled weights — no routing, no extra expressivity. Gating on the
> branch activations makes $\sigma$ a function of $x$, which is what allows
> different inputs to recruit different branches.

### Exact reduction to baseline

With $k=1$, `soma = sum`, $\theta = 0$:

$$x_{\text{sparse}} = \mathrm{ReLU}(x\,E^{\text{den}}), \quad E^{\text{den}} \in \mathbb{R}^{nh \times D \times 1 \times N} \cong E$$

which is *exactly* step 1. Asserted at `atol=0` in
`tests/test_neuron_units.py::test_dendritic_k1_sum_is_exactly_the_default_unit`,
and the parameter count matches baseline exactly. So `k` is a clean knob: at
`k=1` this *is* BDH.

---

## Parameter count

The two encoders grow by `k`; the decoder does **not**, because it reads one
soma output per neuron.

| | baseline | dendritic |
|---|---|---|
| `E`, `E_v` | $2nD$ | $2knD$ |
| `W_dec` | $nD$ | $nD$ |
| total backbone | $3nD$ | $(2k+1)nD$ |

Matched-parameter neuron count:

$$n' = \frac{3n}{2k+1}$$

At `tiny` (`D=128, nh=4, n=4096`): `k=2 → mult 19`, `k=4 → mult 11`,
`k=8 → mult 6`, all within 10% of baseline
(`param_accounting.dendritic_budget`).

**The trade is explicit: `k` branches buy input-specific sub-fields at the cost
of ~3× fewer neurons at `k=4`.**

---

## Why `max` is the variant that could matter for continual learning

With `max`, for each (token, neuron) pair:

$$\frac{\partial\, x_{\text{sparse},j}}{\partial\, b_{c,j}} = \mathbb{1}\big[c = \arg\max_{c'} a_{c',j}\big]\cdot \mathbb{1}\big[a_{c,j} > 0\big]$$

**Gradient reaches exactly one branch per neuron per token.** So if task A
drives branch 1 of neuron *j* and task B drives branch 2, then while training
on task B the weights of branch 1 receive **no gradient** — task A's stored
computation is structurally shielded rather than merely "elsewhere in the
space".

This is the distinction the phase-1 results pointed at. Measured there:
representational *overlap* did not predict forgetting (joint and naive had the
same overlap, 0.868 vs 0.872, and opposite outcomes), so a mechanism that only
*separates* representations attacks the wrong thing. `max`-soma dendrites are
a **protection** mechanism, which is the failure mode the data implicated.

With `sum`, every active branch receives gradient, so there is no shielding —
which is why `sum` is expected to behave much like baseline and is the right
control.

### The caveat

Nothing in the loss *rewards* routing different tasks to different branches.
Under sequential training the model sees one task at a time and has no
incentive to preserve unused branches, so specialisation has to emerge on its
own from input statistics. Whether it does is exactly what the branch probes
measure: `same_branch_frac` (do two tasks drive the same branch of the same
neuron?) against the model's own random-init value, since an untrained network
already scores ~0.65 at `k=4` from weight-norm asymmetry alone.
