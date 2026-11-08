"""Retrain original matched Numenta arm, archive task-associated segment gates."""
import json
import statistics as st
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.matched_numenta_comparison import construct, ROOT
from bdh_cl.scripts.numenta_split_mnist_comparison import rezero_weights,state_hash

@torch.no_grad()
def archive(model,context):
    indices=[];modulation=[]
    for layer in model._layers:
        d=layer.segments(context[None,:])
        ids=d.abs().argmax(-1)
        indices.append(ids.squeeze(0).to(torch.uint8))
        modulation.append(d.gather(2,ids[...,None]).squeeze(0).squeeze(-1).sigmoid())
    return torch.stack(indices),torch.stack(modulation)


def archived_forward(model,x,contexts,taskids,indices,modulations,mode):
    c=contexts[taskids]
    if mode=='native':return model(x,c)
    for i,(layer,activation) in enumerate(zip(model._layers,model._activations)):
        y=layer.module(x)
        if mode=='archived_indices':
            d=layer.segments(c)
            gate=d.gather(2,indices[taskids,i].long()[...,None]).squeeze(-1).sigmoid()
        elif mode=='archived_modulation':gate=modulations[taskids,i]
        else:raise ValueError(mode)
        x=activation(y*gate)
    return model._output_layers[0](x)

@torch.no_grad()
def evaluate(model,x,y,contexts,indices,modulations):
    model.eval();before=state_hash(model)
    ii=indices.clone();mm=modulations.clone();scores={}
    for mode in ['native','archived_indices','archived_modulation']:
        scores[mode]={}
        for router in ['prototype','oracle_task']:
            hits=0
            for start in range(0,len(x),256):
                bx=x[start:start+256];by=y[start:start+256]
                tasks=by//2 if router=='oracle_task' else torch.cdist(bx,contexts).argmin(-1)
                logits=archived_forward(model,bx,contexts,tasks,indices,modulations,mode)
                assert logits.shape[1]==10
                hits+=int((logits.argmax(-1)==by).sum())
            scores[mode][router]=hits/len(x)
    assert state_hash(model)==before and torch.equal(ii,indices) and torch.equal(mm,modulations)
    return scores


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/numenta_archived_prototype_matched_20261101';out.mkdir(parents=True,exist_ok=False)
    protocol=dict(seeds=[101,102,103],orders=['forward','reverse'],learning_rate=.0005,rate_source='Earlier matched Numenta validation-selected setting, no new tuning',epochs_per_task=2,batch_size=128,training='UNMODIFIED original matched Numenta training: native absmax context gating,5%KWinners,continuousAdam,clip5,rezero fixed masks,110k presentations',archive='After each task save per-layer/per-neuron segment indices and, in a separate stronger diagnostic, historical sigmoid modulation amplitudes',inference='Nearest raw-input prototype selects task-associated archive entry, all10classes compete. Native retains original prototype-as-dendritic-input. Index archive overrides segment argmax but recomputes segment response using current dendritic weights and chosen prototype. Modulation archive bypasses dendritic context computation and restores saved amplitudes, with same final feedforward/classifier weights.',limits='Index archive most closely matches storing branch choices, but Numenta still needs context to compute segment modulation. Frozen modulation is a stronger archive that stores historical learned gate values, not merely indices. Neither restores feedforward weights or neuron kWinner masks. Known training boundaries allowed. Three seeds×two orders. Oracle task selection onlydiagnostic.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    raw=load_raw(str(ROOT/'data_cache/mnist'))
    x=(torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=(torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=np.random.default_rng(20261101);vi=np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==c))[:500] for c in range(10)]);ti=np.setdiff1d(np.arange(len(y)),vi)
    np.savez(out/'split_indices.npz',train=ti,validation=vi)
    assert not np.intersect1d(ti,vi).size
    x,y=x[ti],y[ti];ix=[torch.where(y//2==g)[0] for g in range(5)];contexts=torch.stack([x[z].mean(0) for z in ix]);torch.save(contexts,out/'contexts.pt')
    original=json.loads((ROOT/'results/matched_numenta_20261031/results.json').read_text());rows=[]
    for seed in [101,102,103]:
        for name,order in [('forward',list(range(5))),('reverse',list(range(4,-1,-1)))]:
            model=construct('numenta',seed);opt=torch.optim.Adam(model.parameters(),lr=.0005)
            indices=torch.zeros(5,2,64,dtype=torch.uint8);modulations=torch.zeros(5,2,64)
            presentations=0
            for stage,g in enumerate(order):
                model.train()
                for epoch in range(2):
                    gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
                    for ids in ix[g][torch.randperm(len(ix[g]),generator=gen)].split(128):
                        bx,by=x[ids],y[ids];opt.zero_grad(set_to_none=True)
                        loss=F.cross_entropy(model(bx,contexts[g].expand(len(bx),-1)),by)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();model.apply(rezero_weights);presentations+=len(ids)
                indices[g],modulations[g]=archive(model,contexts[g])
            assert presentations==110000
            oldck=torch.load(ROOT/f'results/matched_numenta_20261031/{seed}_{name}_numenta.pt',weights_only=True)['model']
            # Exact final parameters certify training was not modified by archiving.
            exact_prior=all(torch.equal(p,oldck[n]) for n,p in model.named_parameters())
            prior_max_delta=max(float((p.detach()-oldck[n]).abs().max()) for n,p in model.named_parameters())
            scores=evaluate(model,tx,ty,contexts,indices,modulations)
            old=next(z for z in original if z['seed']==seed and z['order']==name and z['arm']=='numenta')
            prior_native_accuracy=old['scores']['inferred']['accuracy']
            modelbytes=sum(v.numel()*v.element_size() for v in model.state_dict().values());ctxbytes=contexts.numel()*contexts.element_size()
            row=dict(seed=seed,order=name,scores=scores,parameters=sum(p.numel() for p in model.parameters()),presentations=presentations,exact_original_parameters=exact_prior,prior_max_parameter_delta=prior_max_delta,prior_native_accuracy=prior_native_accuracy,payload_mb={'native':(modelbytes+ctxbytes)/1e6,'archived_indices':(modelbytes+ctxbytes+indices.numel()*indices.element_size())/1e6,'archived_modulation':(modelbytes+ctxbytes+modulations.numel()*modulations.element_size())/1e6})
            rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));torch.save(dict(model=model.state_dict(),indices=indices,modulations=modulations),out/f'{seed}_{name}.pt')
            print('FINAL',seed,name,scores,flush=True)
    summary=[]
    for mode in ['native','archived_indices','archived_modulation']:
        vals={router:[st.mean(100*z['scores'][mode][router] for z in rows if z['seed']==s) for s in [101,102,103]] for router in ['prototype','oracle_task']}
        summary.append(dict(mode=mode,autonomous=st.mean(vals['prototype']),sd=st.stdev(vals['prototype']),oracle=st.mean(vals['oracle_task']),oracle_sd=st.stdev(vals['oracle_task']),payload_mb=rows[0]['payload_mb'][mode]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Numenta with task-associated configuration archives','','Same original native Numenta training. Three seeds, two orders, same training recipe as the earlier matched study. All inference modes share the SAME newly trained final weights. All10outputs eligible. No replay or restored feedforward weights.','','| Inference | Prototype autonomous ± seed SD | Oracle task selection | Retained MB |','|---|---:|---:|---:|']
    for z in summary:lines.append(f"| {z['mode']} | {z['autonomous']:.2f} ± {z['sd']:.2f}% | {z['oracle']:.2f}% | {z['payload_mb']:.6f} |")
    lines+=['','Index archive stores640uint8segment choices (640bytes). It retrieves those choices by the common nearest-input-prototype router, analogous to the original branch experiment. Numenta segment amplitude still depends on the selected context and CURRENT dendritic weights.','','Frozen-modulation archive is an additional, stronger diagnostic: it saves640FP32sigmoid values (2560bytes), restores those gate amplitudes, and uses the prototype only as an archive lookup key. It stores historical numerical gate values, unlike the branch model’s index-only archive. It does not restore earlier feedforward/classifier parameters or neuron winner sets.','','Known training boundaries are used to associate prototypes with archives. Oracle uses true pair to select archive entry, not output masking. Retained state is numerical payload, excluding optimizers and peak RAM. The earlier context-trace hybrid is a different experiment and is not combined here. Three-seed SD averages orders within seed.','','Checks: native scoring uses the upstream forward; all inference modes share final model parameters; parameter differences from earlier checkpoints recorded; inference state/archives remain unchanged; all10logits eligible;110kpresentations/run; identical split. All archives/checkpoints/source/protocol are retained.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',summary,flush=True)
if __name__=='__main__':main()
