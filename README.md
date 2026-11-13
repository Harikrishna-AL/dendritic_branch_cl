# Dendritic branch routing for continual learning

Harikrishna · status as of 13 November 2026 · exploratory research code

## Objective

Can a network learn tasks one after another, **without replaying old data and without being told the task at test time**, if each neuron has several dendritic branches and learns *which branch to use* through a local competition?

There are two questions:

1. **Retention.** Does choosing branches per neuron protect old tasks better than ordinary networks of the same size, including published protection methods (synapse-level stability and HAT task masks)?
2. **Access.** At test time the network has to pick the right branches from the input alone. How well can it do this from one image, and can a short-lived memory carried across a stream of inputs do better?

## The mechanism

- **Branches.** Each hidden neuron (or conv channel) has 4 dendritic branches. A branch is a tanh unit that sees its own fixed random subset of the inputs; in the MNIST network, 196 of the 784 pixels.
- **Training: branches compete for each neuron.**
  - A neuron outputs one branch at a time.
  - Every branch keeps a running *error trace*: an estimate of what the loss would have been if that branch had driven the neuron.
  - The branch with the lowest trace wins and receives the task loss.
  - A small extra loss term also nudges the other branches, so they stay usable later.
- **Routes.** At the end of each task, every neuron's current winning branch is stored. This list of one branch index per neuron is the task's *route*.
- **Test time.** With no task information, each neuron mixes its branches, weighting each by how strongly it responds (*soft mixing*).

The code is in `branch_routing/models/branch_net.py`.

## Terms used below

**Setting**
- **Split MNIST / Split CIFAR-10:** the 10 classes are split into 5 *tasks* of 2 classes ({0,1}, {2,3}, …). Tasks are learned one after another, with no access to earlier tasks' data.

**Models**
- **Branch network:** the model above. On CIFAR, it has branches in every conv layer and in the top layer.
- **Ordinary network:** a standard MLP or CNN with the same number of parameters and no branches.
- **Ordinary network + synaptic stability:** the ordinary network with the protection rule of Cho et al. (*Neural Networks*, 2025):
  - every weight gets a fixed random "flexibility";
  - weights that are inflexible and have already moved a lot become harder to change;
  - **α** sets how strong this protection is. It was swept from 3 to 30,000, and the best value on validation data is reported.
- **HAT** (Hard Attention to the Task; Serrà et al., ICML 2018), the strongest comparable method:
  - each task learns a mask that switches the network's units on or off;
  - weights used by earlier tasks are frozen;
  - it is ported from the official code, with one shared 10-class head, and has the same parameter count as the branch network;
  - learning rate and dropout are chosen on validation data.

  Unlike the branch network, HAT **must be told the task** to make any prediction.

**Prediction modes**
- **Image only:** the network gets only the image (soft mixing for the branch network). HAT has no such mode.
- **True task:** the model is told the image's real task. The branch network then uses that task's stored route, and HAT uses that task's mask. This is a diagnostic: it shows whether old knowledge is still *stored*, separately from whether it can be *found*.
- **Pixel-key router:** the task is chosen by a per-task statistical summary of the pixels (section 2). The model then predicts within that task's two classes. The same router is used for every model.

**Metrics**
- **All-10 accuracy:** the prediction is among all 10 classes. This is the standard *class-incremental* score.
- **Within-pair accuracy:** the prediction is limited to the two classes of the image's real task (the *task-incremental* score). This measures how much of each task is retained.
- **Learned → kept:** within-pair accuracy on a task right after learning it → at the end of training. Both are on validation data and averaged over the first 4 tasks.
- **Forgetting:** learned minus kept, in percentage points.

**Statistics**
- **±:** standard deviation over 3 seeds. Each seed is averaged over two task orders, forward and reverse.
- **[a, b]:** 95% confidence interval of a paired difference over the 3 seeds.

## Results

Setup: 2 epochs per task, Adam with learning rate 5·10⁻⁴. All numbers are on the test set unless stated otherwise. They were produced by this code with 2 CPU threads, and the full tables are in `results/`.

### 1. Retention: branch network vs other networks of the same size

| | Model | All-10 accuracy, image only | Within-pair accuracy | Learned → kept | Forgetting |
|---|---|---:|---:|---:|---:|
| MNIST | **Branch network** | **69.6 ± 0.4** | 98.5 ± 0.5 (true task) | 98.7 → 98.5 | 0.2 |
| | HAT | needs the task | **98.9 ± 0.1** (true task) | 99.1 → 99.0 | **0.0** |
| | Ordinary network | 20.3 ± 0.0 | 83.8 ± 2.3 | 98.9 → 80.5 | 18.4 |
| | Ordinary network + synaptic stability (α = 1000) | 51.5 ± 2.4 | 93.9 ± 2.3 | 97.3 → 94.4 | 2.9 |
| CIFAR-10 | **Branch network** | **18.7 ± 0.8** | 73.0 ± 0.4 (true task) | 77.4 → 71.4 | 6.0 |
| | HAT | needs the task | **74.1 ± 2.7** (true task) | 72.3 → 72.9 | **−0.6** |
| | Ordinary network | 16.0 ± 0.6 | 58.4 ± 1.8 | 76.9 → 53.1 | 23.8 |
| | Ordinary network + synaptic stability (α = 3) | 16.3 ± 0.1 | 58.2 ± 2.0 | 76.4 → 52.5 | 23.9 |

Within-pair accuracy with the true task, branch network minus each model:

| Compared with | MNIST | CIFAR-10 |
|---|---|---|
| HAT | −0.4 [−1.3, +0.5] | −1.1 [−7.1, +4.9] |
| Synaptic stability | +4.6 [+0.1, +9.1] | +14.8 [+9.7, +20.0] |

Image-only all-10 accuracy vs synaptic stability: +18.2 [+12.8, +23.6] on MNIST, +2.3 [+0.2, +4.5] on CIFAR-10.

**What this shows:**
- **The branch network keeps what it learns.** It learns each task as well as an ordinary network does (98.7% / 77.4%) and loses almost none of it: 0.2 / 6.0 points, against 18.4 / 23.8 for the ordinary network.
- **HAT retains as well as or slightly better than the branch network.** The difference isn't significant on either benchmark, and HAT forgets essentially nothing. So the branch network **does not beat HAT at retention; it ties.**
  - With the same pixel-key router, the two are also equal: 91.0 vs 91.2% on MNIST, 32.8 vs 32.8% on CIFAR-10.
  - What HAT lacks is a mode without task information. It can't predict at all unless a task is given or inferred, whereas the branch network gives 69.6% on MNIST from the image alone.
- **Synaptic stability only trades learning for keeping.** Stronger α reduces forgetting, but the network then learns new tasks less well.
  - **MNIST:** the best setting freezes 93% of the weights.
  - **CIFAR-10:** at the strongest α, forgetting drops to 1.5 points, but new tasks are learned to only 62% instead of 77%. No α improves the final result over the ordinary network, so the best setting (α = 3) is essentially the ordinary network.
- **Branch placement on CIFAR-10.**
  - **Branches in every conv layer** (the model above): forgetting 4.4–6.0 points.
  - **Branches only in the top layer:** forgetting 9.8 points.
  - **Ordinary network:** forgetting 23.8 points.
- **Depth (MNIST).** With 1 / 2 / 3 branch layers, true-route all-10 accuracy stays at 98.5 / 98.0 / 94.5%. Ordinary networks stay at about 20% at every depth.

### 1b. Does the error-trace rule matter? (MNIST)

The same branch network, loss and data are used throughout. Only the rule that picks each neuron's training branch changes:
- **Error trace (ours):** the branch with the lowest running error estimate.
- **Fixed random:** one random branch per neuron, the same for every task.
- **Random per batch:** a new random branch for every batch.
- **Most active:** the branch that responds most strongly to the current batch.
- **Assigned by task (upper bound):** a random branch per neuron for each task, chosen *using the task identity*. This is a perfect task-specific allocation that the error trace has to discover by itself.

| Training rule | All-10 accuracy, image only | All-10 accuracy, true task | Routes differ between tasks |
|---|---:|---:|---:|
| **Error trace** | **69.6 ± 0.4** | **98.5 ± 0.4** | 77% |
| Fixed random | 20.2 ± 0.0 | 20.2 ± 0.0 | 0% |
| Random per batch | 20.2 ± 0.0 | 20.2 ± 0.0 | 75% |
| Most active | 20.4 ± 0.2 | 37.0 ± 2.5 | 50% |
| Assigned by task (upper bound) | 43.1 ± 1.2 | 98.4 ± 0.2 | 74% |

"Routes differ" is the share of neurons whose stored branch differs between two tasks; 75% is the share expected if branches were assigned at random.

**What this shows:**
- **The error trace is what makes retention work.** With random or activity-based selection, the stored routes are useless (20–37%).
- **The error trace reaches the upper bound without using the task identity:** 98.5 vs 98.4% with the true task.
- **Its image-only accuracy is much better than the upper bound's** (69.6 vs 43.1%). Routes chosen by the error trace also make the soft mixing work, and routes assigned at random do not.

With the pixel-key router, every rule scores 89–91%, even the random ones. The router restricts prediction to two classes, and that restriction does most of the work. So the pixel-key column is not used to compare methods here.

### 2. Access: finding the right route from one image

*Task keys.* At the end of each task, the network stores a small statistical summary (a Gaussian) of that task's training images. It is computed from that task's data only, so nothing is replayed. At test time the image is assigned to the task whose summary fits it best, and that task's route makes the prediction. Two kinds of key are compared:
- **Branch-response key:** summarises the branches' responses to the images.
- **Pixel key:** summarises the raw pixels, keeping their 50 main directions of variation (PCA-50).

| How the route is chosen | MNIST all-10 accuracy | MNIST task found | CIFAR-10 all-10 accuracy | CIFAR-10 task found |
|---|---:|---:|---:|---:|
| No choice (soft mixing) | 69.6 | – | 18.7 | – |
| Branch-response key | 27.1 | 27.4 | 18.8 | 23.8 |
| **Pixel key** | **91.0** | **91.9** | **32.8** | **43.1** |
| Real task given (upper bound) | 98.5 | 100 | 73.0 | 100 |

"Task found" is the percentage of test images assigned to their real task. Chance is 20%.

**What this shows:**
- **Branch-response keys fail.** Branches keep changing after their task ends, so a stored summary no longer matches the final network. A key on random combinations of pixels (81% task found) already does better than the branches' own responses.
- **A pixel key works well on MNIST** (91.0% all-10 accuracy), but it is a generic density model, not a branch mechanism.
- **On CIFAR-10, one image carries limited task information.**
  - The same small CNN, trained the *normal* way on all tasks at once, identifies the task of a single image only 61–66% of the time.
  - The pixel key reaches 43% without stored data.
  - So the CIFAR-10 gap is mostly a property of the setting and the network size (`results/taskid_ceiling/`).

### 3. Short-term branch memory over a stream (MNIST)

Test images arrive one at a time, without labels. Three kinds of stream are tested:
- **Blocks of 200:** 200 images from one task, then the next task.
- **Random blocks:** block lengths are random, 50 images on average.
- **Mixed:** tasks in random order, with no structure to exploit.

Methods:
- **No memory:** each image on its own (soft mixing).
- **Branch memory (ours):** each neuron keeps a fading record of which of its branches best explained the network's *own* recent predictions. That record is mixed into how the next image is routed. A *switch-off rule* disables the memory when consecutive predictions stop agreeing on a task, which protects mixed streams.
- **Class-frequency memory (control):** a fading record of which classes were recently predicted, used to bias the next prediction. It gets the same switch-off rule. It tests whether the gain needs branches at all.
- **Branch memory from true labels (upper bound):** like branch memory, but fed the real labels. This is not available in practice.

Each method's settings are chosen on validation data by its average over the three stream types.

| Method | Blocks of 200 | Random blocks | Mixed |
|---|---:|---:|---:|
| No memory | 69.6 ± 0.4 | 69.6 ± 0.4 | 69.6 ± 0.4 |
| Class-frequency memory (control) | 74.5 ± 1.2 | 73.1 ± 1.2 | 69.4 ± 0.4 |
| **Branch memory** | **78.7 ± 1.2** | **77.6 ± 1.0** | **69.5 ± 0.4** |
| Branch memory from true labels (upper bound) | 95.7 ± 0.2 | 93.1 ± 0.2 | 69.1 ± 0.3 |

Branch memory minus class-frequency memory: +4.2 [+3.6, +4.8] on blocks of 200, +4.5 [+4.0, +5.0] on random blocks, +0.1 [−0.0, +0.2] mixed.

**What this shows:** when the context persists, a label-free branch memory adds 9 points over no memory and 4 points over a memory that doesn't use branches. When inputs are mixed it costs nothing.

### 4. Why image-only accuracy is low: diagnostics (MNIST)

These experiments ask where the gap between image-only accuracy (69.6%) and true-route accuracy (98.5%) comes from, and whether simple fixes close it. All are on Split MNIST, so they are exploratory (see the note on the MNIST test set below).

**Depth makes image-only accuracy worse because routing errors pile up, not because the training signal fails to reach early layers** (`results/depth/summary.md`).
- With the true route, accuracy stays high at every depth: 98.5 / 98.0 / 94.5% at 1 / 2 / 3 layers. With soft mixing it falls: 69.6 / 50.0 / 40.3%.
- The training signal is similar in every layer (mean |dCE/dh| 0.0047–0.0048 per layer in the 3-layer network).
- In a 2-layer network, giving the true route to either layer alone recovers 92–97%; soft mixing in both layers gives 50%.
- Giving the last layer another task's route collapses all-10 accuracy (11–16%), while doing so for earlier layers costs little (87–88%). The last layer's route decides the task.
- Sharper soft mixing (a temperature) and picking one whole stored route do not help.

**Branch representations are compressible.** Trained branch layers become far more compressible than at initialisation (a 2-unit autoencoder leaves 13% of the variance unexplained, against 66% untrained), and more than an ordinary network at 1 layer (32%). The task information is there; the soft mixture does not use it (`results/depth/compressibility.md`).

**Training the gate directly fails** (`results/gate_loss/`).
- *Gate loss:* a cross-entropy pushing softmax(|response|) towards the branch that is learning. Every weight tested lowers image-only accuracy (λ = 1: 20.7%), even though route mass (the soft weight on the true route's branch, chance 25%) rises from 31.6 to at most 41%.
  - The reason is recency bias. With no old images there are no negatives, so the newest task's branches become loud on every digit.
- *Calibration:* dividing each route's strength by its value at the end of its own task (the analogue of BiC / weight aligning) gives 51.6%, below soft mixing. Old routes' strength erodes as later tasks are trained.
- *Upper bound, not legal:* the same gate loss with replayed old images as negatives reaches 80.4% at 1 layer (old-image weight 10), 74.7% at 2 and 71.1% at 3 layers. So even a perfectly informed gate of this form leaves about 18 points.

**Global inhibition and local inputs do not help** (`results/kwta/`, `results/patch/`).
- APL-style k-winners-take-all over a layer's branches (as in the fly mushroom body): at test only, at most 65.0%; trained with it, 39–58% at 1 layer and 11.0% at 3 layers. Active branches are enriched for the true route (1.25–2.6× chance), but not enough.
- Each first-layer branch reading one 14×14 patch instead of 196 random pixels: +0.5 [−3.9, +4.8] points at 1 layer, +9.9 [+0.7, +19.1] at 2 layers, −1.0 [−7.1, +5.1] at 3 layers.

**Where the missing accuracy is** (`results/gap_anatomy.md`).
- Almost all errors are picking the wrong task. When the task is right, the class is right 99.4% of the time (1 layer).
- The errors are between look-alike digits from tasks never trained together: 9→4, 0→5, 8→5, 4→9, 7→9.
- A linear probe trained on all tasks jointly (not legal) reaches about 92% on the network's representation, close to 91.1% on raw pixels. So separating look-alike digits from different tasks is limited by the input features, not only by the gating.

### 5. On frozen pretrained features (ImageNet ResNet-18)

If look-alike inputs are the problem, better features should help. Here the branch network (1 layer, 168 neurons × 4 branches, each branch reading 128 of the 512 features) is trained on frozen ImageNet ResNet-18 features, with the same continual protocol. It is compared with two standard no-replay methods for pretrained features:
- **SimpleCIL:** nearest cosine class mean.
- **RanPAC:** a random ReLU projection followed by ridge regression from running sums. This equals joint training of the ridge readout, so it does not forget by construction. Our version has no first-session backbone adaptation.

RanPAC stores no images, but it does keep statistics of all past data (the sums G = Σhhᵀ and C = Σhyᵀ), so it is also compared at matched storage.

| Method | Numbers stored | MNIST | CIFAR-10 |
|---|---:|---:|---:|
| Branch network on raw pixels (sections 1–2) | | 69.6 ± 0.4 | 18.7 ± 0.8 |
| **Branch network, soft mixing (no task info)** | 88,378 + 840 route indices | **88.6 ± 0.4** | **77.7 ± 0.5** |
| Branch network + per-task feature key | 219,193 | 94.8 ± 0.1 | 79.8 ± 0.3 |
| Ordinary MLP, same size | | 38.3 ± 0.2 | 32.6 ± 0.4 |
| SimpleCIL | 5,120 | 77.1 | 70.4 |
| RanPAC, 200-d projection | 42,000 | 92.0 ± 0.7 | 78.7 ± 0.4 |
| RanPAC, 300-d projection (≈ branch network's storage) | 93,000 | 93.9 ± 0.3 | 81.4 ± 0.6 |
| RanPAC, 5,000-d projection (standard) | 25,050,000 | 97.9 ± 0.0 | 88.0 ± 0.1 |
| Ridge on raw 512-d features | 267,264 | 96.7 | 85.7 |
| Linear probe on all tasks jointly (ceiling, not legal) | | 97.9 | 87.3 |

All-10 accuracy, image only. Full tables, including the whole RanPAC size sweep and the confusions, are in `results/backbone/`.

**What this shows:**
- **Good features close most of the gap.** Image-only accuracy rises by 19 points on MNIST and 59 points on CIFAR-10. Forgetting stays near zero (0.0 and 0.3 points), and within-pair accuracy with the true route is 99.3 and 96.6%.
- **The branch network beats SimpleCIL and an ordinary network**, by 7–12 and 45–50 points.
- **It does not beat RanPAC.** At about the same storage, RanPAC is 5.3 points better on MNIST and 3.7 on CIFAR-10. The standard 5,000-d RanPAC is 9–10 points better.
- **Routing itself barely improved.** Route mass is 33.6% (MNIST) and 31.7% (CIFAR-10), close to 31.6% on raw pixels. The gain comes from features on which the soft mixture's errors matter less, not from a better choice of branches.

## Novelty: what is claimed and what is not

**Already known (not claimed):**
- Dendrites can help continual learning: Active Dendrites (Iyer et al. 2022), DendSN (Huang et al. 2024).
- Giving each task its own subnetwork protects old tasks: PackNet, HAT, Supermasks.
- Making individual synapses harder to change protects weights (Cho et al. 2025 and earlier work).

**Candidate contribution:**
- **Task routes come from a local competition, without task identity.** Within each neuron, branches compete through a local error trace, and task routes appear as a side effect of ordinary training. There is no learned mask, pruning step, given context signal or task embedding.
  - This rule is what makes retention work. Random or activity-based selection gives useless routes (section 1b).
  - It matches an allocation that uses the task identity: 98.5 vs 98.4%.
- **Retention equal to HAT, from a different, local mechanism.** HAT learns a mask per task and blocks gradients to protected weights; the branch network uses only each neuron's own error trace. The branch network matches it at the same size (section 1). It is not better.
- **A prediction mode without any task information.** HAT and similar mask methods cannot predict without a task. The branch network can: 69.6% all-10 accuracy on Split MNIST from the image alone, with no replay and no task information.
  - For comparison, an ordinary network gets 20% and synaptic stability 51.5%.
  - Error-trace routes give much better soft mixing than task-assigned ones (69.6 vs 43.1%).
- **A short-term branch memory helps in streams.** It is label-free, beats a matched non-branch memory when context persists, and costs nothing when inputs are mixed (section 3).
- **On frozen pretrained features**, image-only accuracy reaches 88.6% (MNIST) and 77.7% (CIFAR-10) with near-zero forgetting (section 5). This is ahead of SimpleCIL but 4–5 points behind RanPAC at the same storage, so it is not a claim of superiority.

**Not yet shown, and the next tests:**
1. **PackNet and Supermasks** at the same size. HAT is done (section 1).
2. **Stream memory on CIFAR-10:** accumulate the pixel-key evidence over a stream.
3. **Fresh seeds and Fashion-MNIST.** The MNIST test set has been looked at many times during development, so the MNIST numbers are exploratory.
4. **Why old routes survive.** The extra loss term also nudges losing branches, and it is not yet analysed why old routes still work.
5. **Training the backbone.** Section 5 uses frozen features, where RanPAC's exact joint statistics are hard to beat. The open question is whether branch routing can adapt features continually without them drifting, and beat frozen-feature RanPAC where frozen features are weak.

## Earlier comparisons (archived, larger model)

Earlier in the project, a larger branch network (about 529k parameters) was compared with many published methods on Split MNIST under the same split and budget. In those runs the task was picked by the nearest average task image. These runs were **not** repeated with the current code and cannot be pooled with the tables above. All-10 accuracy:

| Group | Method | Accuracy |
|---|---|---:|
| Branch network | 529k parameters, nearest-average-image router | 74.8 |
| Store old images | DER++, 1,000 images / 200 images | 94.5 / 88.4 |
| | ER-ACE, 1,000 / 200 images | 91.1 / 85.7 |
| | OCAR (ICML 2025), adapted | 93.1 |
| Store no images | PEC (ICLR 2024), same size | **91.4** |
| | HAT, same router | 75.2 |
| | RFE (TMLR 2026), adapted | 59.3 |
| | Online EWC / Synaptic Intelligence | 23.4 / 20.3 |
| | Numenta Active Dendrites | 18.6 |
| Statistical models on PCA-50 pixels | Streaming QDA / Streaming LDA | 95.2 / 86.1 |
| Offline reference | Joint training on all data | 97.0 |

**What this shows:**
- **For single-image accuracy, stronger no-replay methods exist.** PEC and simple statistical classifiers beat the branch networks, consistent with section 2.
- **The branch networks clearly beat regularisation methods** (EWC, SI) and Active Dendrites.

Sources: `archive/reports/continual_learning_evidence_20261031/`, `archive/reports/leaderboard_constraints_20261031/`, `archive/reports/recent_cl_comparison_20261031/`, `archive/results/rawpixel_gating_baselines_20261101/`.

## Limitations

- Small networks (about 134k parameters on MNIST, 160k on CIFAR-10), 3 seeds, 2 epochs per task.
- Task boundaries are known during training.
- Image-only all-10 accuracy on CIFAR-10 is only slightly above the ordinary network (+2.3 points).
- On MNIST, image-only accuracy falls as branch layers are added: 69.6 → 50.0 → 40.3% at 1 / 2 / 3 layers (section 4).
- Sections 4 and 5 were run after looking at the MNIST test set many times; they are diagnostics, not final evaluations.
- CPU results depend on the number of threads, so all runs use 2.

## Code layout

```
branch_routing/                 library
  data.py                       Split MNIST / Split CIFAR-10 loaders (task = label // 2)
  models/layers.py              SparseBranchLayer, ConvBranchLayer
  models/branch_net.py          BranchNetwork (training rule), BranchMLP, BranchCNN
  models/baselines.py           ordinary MLP and CNN, parameter matching
  models/hat.py                 HAT networks (MLP and CNN)
  continual/trainer.py          ContinualTrainer: task-by-task training, route storage
  continual/metaplasticity.py   synaptic stability control (Cho et al. 2025)
  continual/hat.py              HAT training loop (official recipe)
  continual/gate_rules.py       ablation: other ways to pick the training branch
  continual/evaluation.py       all-10 / within-pair / true-route metrics
  access/keys.py                GaussianKey, TaskKeyRouter (task keys)
  access/stream.py              stream orders, BranchState, ClassPriorState, PersistenceGate
  stats.py                      seed means, paired intervals
scripts/                        entry points (build objects, run, save JSON)
  train.py                      one configuration -> results/runs/<bench>/*.json
  run_sweep.py                  every configuration behind section 1 (one process per run)
  summarize.py                  -> results/summary_<bench>.md (incl. the full α sweep)
  summarize_baselines.py        HAT + gate ablation -> results/summary_baselines.md
  task_keys.py                  section 2 -> results/task_keys/
  taskid_ceiling.py             section 2 ceiling -> results/taskid_ceiling/
  stream_state.py               section 3 -> results/stream_state/
  depth_diagnostic.py           section 4, depth -> results/depth/
  compressibility.py            section 4, compressibility -> results/depth/compressibility.md
  route_calibration.py          section 4, calibration -> results/gate_loss/calibration*
  gate_upper_bound.py           section 4, upper bound with replay -> results/gate_loss/upper_bound*
  kwta_test.py                  section 4, k-winners-take-all -> results/kwta/
  gap_anatomy.py                section 4, error anatomy -> results/gap_anatomy.md
  backbone_test.py              section 5, frozen ResNet-18 features -> results/backbone/
tests/                          fast interface tests (python -m pytest tests)
data/splits/                    fixed MNIST train/validation split
archive/                        earlier experiments and reports (see archive/README.md)
```

In the code and the raw result files: `cil` = all-10 accuracy, `til` = within-pair accuracy, `soft` = image only, `route` = true route, `plain` = ordinary network, `alpha` > 0 = synaptic stability, `hat` = HAT, `key_routed` = pixel-key router, `most_active` / `fixed_random` / `random_per_batch` / `task_assigned` = gate rules, `class_prior` = class-frequency memory, `branch_state` = branch memory, `_gated` = with the switch-off rule.

## Reproducing

Python 3.13, PyTorch 2.8. Run from the repository root.

```bash
pip install -r requirements.txt
bash scripts/fetch_mnist.sh
python -c "from branch_routing.data import load_cifar10; load_cifar10(download=True)"

python scripts/run_sweep.py --bench mnist --workers 2      # section 1 (~30 min)
python scripts/run_sweep.py --bench cifar10 --workers 3    # section 1 (~1.5 h on a laptop CPU)
python scripts/summarize.py --bench mnist
python scripts/summarize.py --bench cifar10
python scripts/run_sweep.py --bench mnist --suite hat      # section 1, HAT (lr x dropout grid)
python scripts/run_sweep.py --bench cifar10 --suite hat --workers 3
python scripts/run_sweep.py --bench mnist --suite gates    # section 1b
python scripts/summarize_baselines.py

python scripts/task_keys.py --bench mnist                  # section 2
python scripts/task_keys.py --bench cifar10
python scripts/taskid_ceiling.py --target class
python scripts/stream_state.py                             # section 3

# section 4 (Split MNIST). Checkpoints: train.py --save-checkpoints <dir>
python scripts/train.py --bench mnist --model branch --arch 3 --save-checkpoints results/depth/ckpt
python scripts/depth_diagnostic.py
python scripts/compressibility.py
python scripts/train.py --bench mnist --model branch --gate-loss 0.1 --gate-mode full   # also --kwta 0.1, --input-layout patch
python scripts/route_calibration.py run --depth 1 --gate-loss 0 --seed 101 --first 0
python scripts/route_calibration.py summarize
python scripts/gate_upper_bound.py run --depth 1 --cur 1 --old 10 --seed 101 --first 0
python scripts/gate_upper_bound.py summarize
python scripts/kwta_test.py
python scripts/gap_anatomy.py

# section 5 (needs torchvision ResNet-18 weights; features cached in data_cache/features/)
python scripts/backbone_test.py extract --bench cifar10
python scripts/backbone_test.py run --bench cifar10 --model branch --seed 201 --first 0
python scripts/backbone_test.py baselines --bench cifar10
python scripts/backbone_test.py summarize --bench cifar10
```

A single run is `python scripts/train.py --bench mnist --model branch --seeds 101 --orders 0` (about 20 s).

On some macOS machines, long-running CPU PyTorch processes occasionally hang inside the BLAS thread pool. `run_sweep.py` therefore runs each configuration in its own process and kills and retries any that hang.
