# Branches inside the conv layers: does it stop CNN forgetting?

Exploratory, 6 November 2026. Split CIFAR-10 (5 two-class tasks), from scratch, 2 epochs per task, Adam .0005, batch 64, clip 5; seeds 201–203 × forward/reverse; ± is SD over seed means; paired intervals are 95% t over the 3 seeds. No tuning: one fixed recipe.

Every conv channel has 4 filter sets (branches) followed by tanh. During training, each channel uses one branch, chosen by an EMA(.95) of estimated substitution loss: CE + Σ_hw g·(z_branch − h), with g the per-example autograd gradient. Image-only inference mixes branches by softmax(spatial-mean |response|). The oracle restores the archived route at every layer (diagnostic, requires the true task). Top: LayerNorm → sparse 64×4 branch layer → head. Loss = 1.5 CE + .5 mean auxiliary first-order substitution term.

| Arm | Conv layers | Params | Acquisition | Image-only | Oracle | Acquisition − oracle |
|---|---|---:|---:|---:|---:|---:|
| conv_b1_wide (control) | ordinary tanh, 32/64/128 channels | 159,690 | 81.3 | 23.48 ± 0.72 | 61.09 ± 4.79 | 20.2 |
| conv_b4 | 4 branches/channel, 16/32/64 | 160,778 | 78.1 | 20.02 ± 2.35 | 68.39 ± 4.66 | 9.7 |
| conv_b4_sparse | as conv_b4, layers 2–3 branches see 50% of input channels | 160,778* | 77.3 | 19.92 ± 0.94 | 71.56 ± 0.52 | **5.8** |

\*Allocated parameters; half of the layer-2/3 conv weights are masked to zero.

Reference from `cnn_failure_diagnosis_20261106` (ReLU CNN, top branch layer only, same protocol): CNN trained on every task, loss 17.3 (oracle 68.35); CNN frozen after task 1, loss 6.0 (oracle 76.13).

Paired vs the capacity control: conv_b4_sparse forgetting −14.48 pp [−26.66, −2.30], oracle +10.47 [−2.67, +23.61], image-only −3.56 [−7.35, +0.23]. conv_b4 forgetting −10.53 [−33.19, +12.12] (not resolved with 3 seeds).

Neurons/channels whose selected branch differs between two task routes (75% = random assignment):

| Arm | conv1 | conv2 | conv3 | top |
|---|---:|---:|---:|---:|
| conv_b4 | 64.1% | 71.1% | 70.3% | 77.1% |
| conv_b4_sparse | 57.3% | 69.5% | 70.3% | 76.9% |

## Interpretation

- **Branching the conv layers does reduce CNN forgetting.** With input-sparse branches, the learn-to-oracle loss falls from 20.2 points (capacity-matched ordinary CNN) to 5.8, about the same as freezing the CNN entirely (6.0), while the conv layers keep learning on every task. This is the only arm where the interval excludes zero.
- **The early-layer concern is real but partial.** Task routes overlap most in conv1 (57–64% distinct vs 75% for random assignment), as expected for filters useful to every task. Overlap still leaves most channels task-specific.
- **Cost: weaker learning.** Acquisition drops 3–4 points (81.3 → 77–78), likely because each task uses one quarter of the filters (16/32/64 active vs 32/64/128 in the control). Oracle accuracy (71.6%) is therefore still below the frozen-CNN reference (76.1%).
- **Access is unchanged.** Image-only accuracy stays at ~20% in all arms. Branched convs address retention, not task recognition from a single image.

Code: `../../bdh_cl/scripts/branched_conv_cifar.py` (run once per `--arm`).
