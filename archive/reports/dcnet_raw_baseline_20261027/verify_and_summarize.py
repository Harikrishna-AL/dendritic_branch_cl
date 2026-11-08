"""Post-run audit and comparison for the raw-pixel DCNet architectural ablation."""
import io,json,os,statistics as st,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'external/DCNet'))
from bdh_cl.scripts.dcnet_official_baseline import predict,digest,count_bytes
from common.common import parse_args
from bdh_cl.scripts.dcnet_raw_branches import RawBranchDCNet


def main():
    torch.set_num_threads(2);out=ROOT/'results/dcnet_raw_short_20261027'
    assert json.loads((out/'completion.json').read_text())['runs']==3
    rows=json.loads((out/'results.json').read_text());assert len(rows)==3
    os.chdir(ROOT/'external/DCNet')
    torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
    checks={}
    for r in rows:
        seed=r['seed'];p=parse_args(default=True);p.n_tasks=10;p.seed=seed
        m=RawBranchDCNet(p,seed).cpu();state=torch.load(out/f'seed{seed}_latest.pt',map_location='cpu',weights_only=True);m.load_state_dict(state['model']);m.eval()
        assert state['stages']==10 and len(state['history'])==10
        assert not any(isinstance(v,torch.nn.Conv2d) for v in m.modules())
        assert all(torch.isfinite(v).all() for v in m.state_dict().values() if v.is_floating_point())
        before=digest(m);x=torch.rand(11,3,32,32)
        with torch.no_grad():
            score=predict(m,x,10,p.smax);independent=[]
            for t in range(10):
                values=[]
                for k in range(4):
                    rotated=torch.rot90(x,k,(2,3))
                    raw=torch.tanh(m.first(rotated.flatten(1)-.5)).reshape(-1,128,10)
                    h=(raw*torch.sigmoid(p.smax*m.ec0[t]).reshape(1,128,10)).sum(-1)
                    values.append(m.joint_distribution_layer[t](h)[:,k*10:(k+1)*10])
                independent.append(torch.stack(values).mean(0))
            expected=torch.cat(independent,1)
        assert score.shape==(11,100) and torch.isfinite(score).all()
        assert torch.allclose(score,expected,atol=1e-6,rtol=1e-5)
        assert digest(m)==before
        assert all(torch.isfinite(v).all() for v in state['p_mask'].values())
        checks[str(seed)]={'no_CNN':True,'finite_checkpoint':True,'all100_scores_match_independent_rotation_ensemble':True,'state_unchanged_during_inference':True}
    (out/'postrun_verification.json').write_text(json.dumps(checks,indent=2))
    final=[r['final_accuracy']*100 for r in rows];avg=[r['average_incremental']*100 for r in rows];oracle=[r['history'][-1]['oracle_task_diagnostic']*100 for r in rows]
    summary={'final_mean_percent':st.mean(final),'final_sd_percent':st.stdev(final),'average_incremental_mean_percent':st.mean(avg),'average_incremental_sd_percent':st.stdev(avg),'oracle_mean_percent':st.mean(oracle),'continuation_bytes':rows[0]['continuation_bytes'],'model_bytes':rows[0]['model_bytes'],'seeds':[r['seed'] for r in rows]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    report=ROOT/'reports/dcnet_raw_baseline_20261027/REPORT.md';s=report.read_text();s=s.replace('DCNet, 3 completed seeds','Raw-pixel DCNet variant, 3 completed seeds')
    s+='\n## Completed-run audit\n\n'
    s+=f"Three seeds completed. Incremental average **{st.mean(avg):.2f}% ± {st.stdev(avg):.2f}**; final **{st.mean(final):.2f}% ± {st.stdev(final):.2f}**; correct-task oracle diagnostic **{st.mean(oracle):.2f}%** (not the inference-ID-free result). All checkpoints passed finite-state, no-CNN, all-100-class scoring, independent rotation-ensemble and inference-invariance checks.\n\n"
    s+='The same first-layer shape and paired initialization remove the CNN-versus-raw-pixel difference. This remains a DCNet architectural ablation: HAT sigmoid gates may activate multiple branches, native projection/learned heads differ, and DCNet uses 5 feature plus 5 classifier epochs with rotation/colour/crop augmentation. The branch pilot used 5 epochs without image augmentation. DCNet sees approximately 60 augmented views per original example across the two phases, versus five presentations in the pilot; the DAC statistics pass is additional. These results cannot establish an equal-compute algorithm ranking or reproduce the published DCNet score.\n\n'
    s+='[Post-run verification](../../results/dcnet_raw_short_20261027/postrun_verification.json) · [Summary](../../results/dcnet_raw_short_20261027/summary.json)\n'
    report.write_text(s);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
