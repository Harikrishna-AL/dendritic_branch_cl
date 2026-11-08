# Why the branch recipe fails on Split CIFAR-10 CNNs

Exploratory, 6 November 2026. Split CIFAR-10 (5 two-class tasks), from scratch, no augmentation, 2 epochs per task, Adam .0005, batch 64, clip 5. Seeds 201–203 × forward/reverse order; ± is SD over seed means. Branch arms use the MNIST recipe: hard error-trace (EMA .95) branch selection during training, CE + .5 single-neuron substitution CE, soft-magnitude image-only inference. Oracle = archived route plus true task (diagnostic only). Acquisition = validation accuracy on each task right after learning it, all 10 classes eligible. No hyper-parameter selection; one fixed recipe per arm.

## How the previous CNN extension was built

`dendritic_architecture_screen.py` / `cifar10_routing_gap.py`: an ordinary 3-layer stride-2 CNN (16/32/64 channels) trained by backprop on every task, LayerNorm on its 1,024 outputs, then one branch layer (64×4 branches, 256 random features per branch) and a shared head. Branches exist only after the CNN; the CNN itself has no branches, error traces or protection. Image-only inference used hard strongest-branch selection.

## Results

| Arm | What is below the branches | Acquisition | Image-only (soft) | Oracle | Acquisition − oracle |
|---|---|---:|---:|---:|---:|
| cnn_plain | ordinary CNN, no branches | 85.6 | 17.67 ± 0.54 | – | – |
| cnn_learned | CNN trained on every task (previous design) | 85.7 | 22.78 ± 2.05 | 68.35 ± 2.32 | 17.3 |
| cnn_freeze_after1 | CNN trained on task 1, then frozen | 82.1 | 22.22 ± 1.97 | 76.13 ± 2.16 | 6.0 |
| cnn_random | random CNN, never trained | 79.6 | 19.85 ± 2.89 | 72.72 ± 3.26 | 6.9 |
| pixel_sparse | no CNN; 168×4 branches on raw pixels, 768/3072 per branch | 79.3 | 24.88 ± 1.66 | 60.74 ± 1.70 | 18.6 |

Route diagnostics (forward order, seeds 201–202, `route_diagnostics.json`):

| Arm | Neurons whose branch differs between two task routes | Oracle, all 10 classes | Oracle, within true pair |
|---|---:|---:|---:|
| cnn_learned | 77.0% | 74.7% | 75.5% |
| cnn_random | 77.4% | 71.6% | 71.6% |
| pixel_sparse | 70.7% | 60.3% | 74.9% |

## Diagnosis

1. **Two failures, not one.** On MNIST only the image-only access failed (98% oracle → 68%). On CIFAR, retention also fails (acquisition 86% → oracle 68%), and access is far worse (oracle 68% → 23%).
2. **Retention failure is mostly the shared CNN drifting.** Freezing the encoder after task 1 cuts the acquisition-to-oracle loss from 17.3 to 6.0 points; a never-trained random encoder gives 6.9. The branches cannot protect a representation computed below them. The remaining ~6 points are lost inside the branch layer and head, even though routes are as distinct as on MNIST (≈77% vs 78%).
3. **Access failure remains even without drift.** With a frozen encoder and a 76% oracle, image-only accuracy stays at 22%, barely above the plain CNN's 17.7%. Picking branches by response magnitude identifies the task well on MNIST, where digit shapes light up different pixel subsets, but not on CIFAR's two-class natural-image tasks. This is an interpretation; it is not directly tested here.
4. **Raw-pixel branches add head competition.** Under the true route, accuracy within the correct pair is 74.9%, but only 60.3% with all ten classes eligible. Classes from other tasks steal predictions even under the right configuration, a confusion absent in the CNN arms.

## Limits

Short schedule (2 epochs/task), small CNN, no augmentation, one fixed learning rate, three seeds. The previously reported PEC result on this protocol family (33.77%) shows from-scratch replay-free Split CIFAR-10 is hard for every method tested, not only this one.

Code: `../../bdh_cl/scripts/cnn_failure_diagnosis.py`, `../../bdh_cl/scripts/cnn_failure_routes.py`.
