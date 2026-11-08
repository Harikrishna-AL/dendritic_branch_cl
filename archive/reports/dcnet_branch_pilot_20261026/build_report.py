"""Verify completed checkpoints and build the pilot report from saved results."""
import json
import statistics as st
from pathlib import Path
import torch
from bdh_cl.scripts.dcnet_branch_pilot import BranchBank

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'results/dcnet_branch_pilot_fixednorm_20261026'
OUT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    assert json.loads((P/'completion.json').read_text())['complete']
    files=sorted(P.glob('*.pt')); assert len(files)==6
    for file in files:
        state=torch.load(file,weights_only=True)
        assert torch.equal(state['mean'],torch.full((3072,),.5))
        m=BranchBank(int(file.name.split('_')[0]));m.load_state_dict(state['model']);m.stages=state['stages']
        x=torch.randn(13,3072);before={k:v.clone() for k,v in m.state_dict().items()}
        with torch.no_grad():
            actual=m.scores(x)
            for task in range(10):
                ids=torch.arange(m.width)*m.tasks+m.routes[task]
                h=torch.nn.functional.linear(x,m.first.weight[ids],m.first.bias[ids]).tanh()
                expected=torch.nn.functional.normalize(h,dim=-1)@m.codes[10*task:10*(task+1)].T
                assert torch.allclose(actual[:,task],expected,atol=2e-6)
            pred=m.predict(x)
        assert pred.shape==(13,) and pred.min()>=0 and pred.max()<100
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    (P/'postrun_verification.json').write_text(json.dumps({'all_six_checkpoints':True,'all_ten_routes_match_independent_linear_forward':True,'all100_classes_compete':True,'inference_state_unchanged':True,'data_independent_centering':True},indent=2))
    rows=json.loads((P/'results.json').read_text());rr=[r for r in rows if r['arm']=='ioe_dac']
    lines=['# DCNet-inspired branch competition pilot — Split CIFAR-100','',
    'Completed 26 October 2026. This is a raw-pixel branch-model adaptation, not a reproduction of DCNet or a matched SOTA comparison. Six sequential runs: three seeds for each of two fixed loss conditions. No test-set selection.','',
    '## Results','',
    'Percentages; ± is sample standard deviation across initialization/shuffle seeds with one fixed class order. Incremental average is mean seen-class accuracy after each stage; final accuracy evaluates all 100 classes after stage ten.','',
    '| Training | Inference | Incremental average | Final accuracy |','|---|---|---:|---:|']
    for arm in ['ioe_dac','ioe']:
        rs=[r for r in rows if r['arm']==arm]
        for rule in ['head_probability','cosine']:
            a=[100*r['average_incremental'][rule] for r in rs];f=[100*r['final'][rule]['accuracy'] for r in rs]
            lines.append(f'| {arm} | {rule} | {st.mean(a):.2f} ± {st.stdev(a):.2f} | {st.mean(f):.2f} ± {st.stdev(f):.2f} |')
    oracle=[100*r['final']['oracle_task_diagnostic'] for r in rr]
    route=[100*r['final']['head_probability']['routing_accuracy'] for r in rr]
    lines += ['',f'Correct-task oracle diagnostic: **{st.mean(oracle):.2f}% ± {st.stdev(oracle):.2f}**. This is a separately labelled diagnostic, not no-task-ID accuracy. Autonomous task assignment from the winning head is {st.mean(route):.2f}% on average. The gap identifies cross-head competition as a major bottleneck; modest within-task accuracy also limits the model.','',
    '## Implementation','',
    '- Known training boundaries define ten successive class blocks. Only the current block enters updates. No retained old images, individual features, generated examples or joint calibration.',
    '- Raw RGB pixels feed 128 neuron groups with ten tanh branches each. Current-task response energy selects one unused branch per group at the boundary. Train those rows, then commit them. Earlier committed rows stay exactly unchanged; all ten tasks fit without branch reuse.',
    '- Fixed random orthonormal 128-dimensional class targets supervise normalised embeddings. Both arms use current-head cross entropy at temperature 0.1 plus squared distance to the target. `ioe_dac` additionally uses supervised contrastive compaction after epoch one, with temperature adjusted using historical concentration.',
    '- Primary inference concatenates task-head softmax probabilities and selects the highest-scoring class. Secondary inference compares raw cosine scores. Both rules were fixed before testing. Prediction receives only inputs, never true task IDs or labels; every seen class competes.',
    '', '## Protocol and resources','',
    'Official cached CIFAR-100 archive and existing loader; class permutation seed 2026. 45,000 fit images, 5,000 unused validation images, 10,000 official test images. Five epochs per block, Adam 0.001, batch 128, seeds 701/702/703. No pretraining or augmentation. Input preprocessing is fixed x/255 - 0.5, with no fitting or future-task lookahead.', '',
    f'Persistent state: **{rr[0]["persistent_bytes"]:,} bytes = {rr[0]["persistent_bytes"]/1e6:.3f} MB**. Includes the whole preallocated branch bank, targets, routes, flags, concentration statistics and input centering. This is not peak training memory: temporary extracted layer, Adam state, verification snapshots, activations and harness data cost more. Total wall time across six runs, including evaluation: {sum(r["seconds"] for r in rows):.1f} seconds.', '',
    '## Differences from DCNet','',
    'DCNet uses HAT masks, an incremental basis generator, its own compensation formula, rotation augmentation, and a ResNet/long-training recipe. This pilot substitutes hard disjoint branch allocation, QR targets and simpler compaction. Routes are selected from current-task responses, not learned HAT masks. Capacity is preallocated for ten tasks; no branch reuse or efficient shared representation is demonstrated. The 128×10 architecture is larger than earlier 24×4 experiments. Its result cannot be compared directly with the published 65.40% DCNet result.', '',
    'The pilot establishes a concrete task-aware/no-replay/no-inference-ID evaluation, but performs poorly. It does not establish the score of full DCNet, or show that the graph approach cannot work. Compaction gives little primary-score improvement in this recipe.', '',
    'An earlier exploratory run used full-fit-set centering statistics. Its scores are superseded by the fixed-centering runs reported here; its files remain archived separately.', '',
    '## Verification and files','',
    'Passed checks: orthogonal targets, disjoint allocation, exact preservation of committed rows, and evaluation state invariance. Independently reconstructed all ten routes in all six saved checkpoints and matched vectorised inference. Verified fixed centering in each checkpoint.','',
    '[Source](../../bdh_cl/scripts/dcnet_branch_pilot.py) · [Protocol](../../results/dcnet_branch_pilot_fixednorm_20261026/protocol.json) · [Stage results](../../results/dcnet_branch_pilot_fixednorm_20261026/results.json) · [Summary](../../results/dcnet_branch_pilot_fixednorm_20261026/summary.json) · [Verification](../../results/dcnet_branch_pilot_fixednorm_20261026/postrun_verification.json)','',
    '[DCNet original paper](https://ijcai.org/proceedings/2025/715).']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:18])); print('Oracle:',st.mean(oracle));print('Verified six checkpoints; report saved.')

if __name__=='__main__': main()
