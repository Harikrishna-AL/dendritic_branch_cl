"""Frozen-checkpoint protected top-two feedback on the strong existing backbones."""
import hashlib
import json
import math
import statistics as st
from pathlib import Path

import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts import complementary_exchange as base

OUT=Path('results/protected_feedback_20261022')
SOURCE=Path('results/complementary_exchange_20261021')
ARMS=['grouped_shared__complementary__self','grouped_shared__complementary__neighbor','global_independent__complementary__self']
THRESHOLDS=[0.,.25,.5,1.,2.,4.,8.,None]  # None means always-on, JSON-safe infinity.
MODES=['unprotected','top2_only','protected']


def save(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)


@torch.no_grad()
def forward_candidates(m,mem,x,mode):
    raw=torch.cat([m.response(b) for b in x.split(512)])
    q=torch.cat([base.keys(b) for b in x.split(512)])
    score=(q@mem.coefficients(mem.mask_cross,.1).T).reshape(-1,24,4)
    original=m.select(score.flatten(1)).reshape(-1,24,4)
    first=m.classify(raw,original.flatten(1))
    top=first.topk(2,-1)
    classes=top.indices
    margin=top.values[:,0]-top.values[:,1]
    w=m.head.weight
    if mode in ['protected','mixed_normalized']:w=F.normalize(w,dim=1)
    expanded=w[:,:,None].expand(-1,24,4) if m.shared else w.reshape(10,24,4)
    weights=expanded[classes]
    if mode.startswith('mixed_'):
        # A shared top-two signal; normalization can change relative class weights.
        weights=(weights*top.values.softmax(-1)[:,:,None,None]).sum(1,keepdim=True)
    contribution=raw.reshape(-1,1,24,4)*weights
    if mode!='unprotected':
        allowed=torch.zeros_like(score,dtype=torch.bool).scatter_(-1,score.topk(2,-1).indices,True)
        contribution=contribution.masked_fill(~allowed[:,None],-torch.inf)
    proposed=F.one_hot(contribution.argmax(-1),4).float()
    if m.structure!='grouped_shared':
        # Preserve the ordinary model's exact global selection pattern/count.
        # Only a singleton group can swap one selected neuron for another.
        singleton=original.sum(-1)==1
        proposed=torch.where(singleton[:,None,:,None],proposed,original[:,None])
    assert (proposed.sum((-1,-2))==24).all()
    count=proposed.shape[1]
    second=m.classify(raw[:,None].expand(-1,count,-1).reshape(-1,96),proposed.flatten(-2).reshape(-1,96)).reshape(-1,count,10)
    if mode.startswith('mixed_'):selected_logits=second[:,0]
    else:
        candidate=second.log_softmax(-1).gather(-1,classes[:,:,None]).squeeze(-1).argmax(-1)
        selected_logits=second[torch.arange(len(x)),candidate]
    return {'first':first.argmax(-1),'second':selected_logits.argmax(-1),'margin':margin,
            'classes':classes,'proposed':proposed,'original':original}


def metrics(pred,y,order):
    old=~torch.isin(y,torch.tensor(order[-1]))
    return {'accuracy':float((pred==y).float().mean()),'old_accuracy':float((pred[old]==y[old]).float().mean()),
            'new_accuracy':float((pred[~old]==y[~old]).float().mean())}


@torch.no_grad()
def evaluate(m,mem,x,y,order,settings):
    before=base.digest(m,mem)
    cache={mode:[] for mode in {s['mode'] for s in settings}}
    for mode in cache:
        for batch in x.split(512):cache[mode].append(forward_candidates(m,mem,batch,mode))
        cache[mode]={key:torch.cat([r[key] for r in cache[mode]]) for key in ['first','second','margin','classes']}
    rows=[]
    for s in settings:
        c=cache[s['mode']]
        gate=torch.ones(len(y),dtype=torch.bool) if s['tau'] is None else c['margin']<s['tau']
        pred=torch.where(gate,c['second'],c['first'])
        a=c['first']==y;b=pred==y
        rows.append({**s,**metrics(pred,y,order),
                     'trigger_rate':float(gate.float().mean()),
                     'wrong_to_correct':float((~a&b).float().mean()),
                     'correct_to_wrong':float((a&~b).float().mean()),
                     'prediction_change_rate':float((pred!=c['first']).float().mean()),
                     'first_top2_true_class_coverage':float((c['classes']==y[:,None]).any(-1).float().mean()),
                     'before_accuracy':float(a.float().mean())})
    # Row normalization is a positive class-wise scaling; it cannot change branch ranks.
    if 'protected' in cache and 'top2_only' in cache:
        assert torch.equal(cache['protected']['second'],cache['top2_only']['second'])
    assert base.digest(m,mem)==before
    return rows


def all_settings():
    settings=[{'mode':'unprotected','tau':None}]
    for mode in ['top2_only','protected']:
        settings.extend({'mode':mode,'tau':t} for t in THRESHOLDS)
    return settings


def verify():
    checks={}
    x=torch.rand(9,784)
    for arm in ARMS:
        m=base.Model(1,arm);mem=base.Memory();q=base.keys(x)
        mem.cov=q.T@q
        mem.mask_cross=torch.randn(96,64)
        a=forward_candidates(m,mem,x,'protected')
        b=forward_candidates(m,mem,x,'top2_only')
        assert torch.equal(a['proposed'],b['proposed'])
        assert (a['proposed'].sum((-1,-2))==24).all()
        score=(q@mem.coefficients(mem.mask_cross,.1).T).reshape(-1,24,4)
        allowed=torch.zeros_like(score,dtype=torch.bool).scatter_(-1,score.topk(2,-1).indices,True)
        if m.structure=='grouped_shared':assert not (a['proposed'].bool()&~allowed[:,None]).any()
        else:
            singleton=a['original'].sum(-1)==1
            assert not ((a['proposed'].bool()&~allowed[:,None])&singleton[:,None,:,None]).any()
            assert torch.equal(a['proposed'].sum(-1),a['original'].sum(-1)[:,None].expand(-1,2,-1))
    checks.update(row_normalization_does_not_change_route_choices=True,restricted_choices_follow_bottomup_top2=True,
                  active_count_preserved=True,ordinary_per_group_active_counts_preserved=True)
    return checks


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    save('verification.json',verify())
    protocol={'question':'Does row-normalized, branch-top2-restricted, low-margin-gated class feedback improve the strong existing backbones beyond85.53%?',
        'arms':ARMS,'source':str(SOURCE),'checkpoints':'Reuse601-610forward/reversefinalmodelsandonlineassociationstates. No confirmation neural retraining or compactpatchsubstitution.',
        'baseline':'Originalridge maskpredictor .1 andAdam.001 backbones. Groupedself78.75%,oracle95.60;groupedneighbor78.79%,oracle96.26;ordinaryself81.30%. Baselineexactreproductionasserted.',
        'feedback':'Pass1top2classes; classrowL2normalizedoutputweight timescurrentbranchresponse ranks branchchoices. Groupedmodels selectone branch pergroup restricted to top2ridge mask scores. Recomputeall10logits; choosecandidate routebycandidateclasslogsoftmax; finalpredictionunrestricted10classargmax. No routebank/testlabels/statewrites.',
        'ordinary_control':'Retainexact24active and originalpergroupcounts. Singletonactivegroups may swap to one oftop2retrievalalternatives. Empty/multi-activegroups unchanged; notconverted togroupedonewinnerarchitecture.',
        'gate':'Rawfirstpass top1-top2logitmargin <tau. Grid0/.25/.5/1/2/4/8/always. Tau0 exactlybaseline. Tau selectedpermodel byvalidationallclassaccuracy,smallesttau tiebreak. No testselection.',
        'calibration':'Reproduce originaldev60050kfit/10kvalidationrun for3armsatfixedLR.001. Calibratefeedbackonly on validation. Originaldevbaselinereproductionasserted. Savefrozenselectionbeforeofficialtestfeedback.',
        'ablations':'Unprotectedall4branchesalwaysfeedback;top2restrictedwithoutrow-normalization;protectedrow-normalized+top2forallthresholds. All settingssaved,onlyvalidationchosen threshold claimed.',
        'statistics':'10pairedseedmeans,orderaveraged,95%tintervaldf9. Existingconfirmationcohortisreanalysis,notindependentreplication. No newdatasets.',
        'scope':'Only requested high-signal inference diagnostic now. Randomprojectioncompression conditional on crossing85.53%; prototypeweight redesign not tested here. Normalization alone cannot alter perclassbranchargmax.'}
    save('protocol.json',protocol);(OUT/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen=torch.Generator().manual_seed(600)
    valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
    original_dev=json.loads((SOURCE/'development.json').read_text())
    selection={};dev=[]
    for arm in ARMS:
        m,mem,bank,acq,seconds=base.train(600,arm,.001,x[mask],y[mask],x[valid],y[valid],SPLITS)
        rr=evaluate(m,mem,x[valid],y[valid],SPLITS,all_settings())
        old=next(r for r in original_dev if r['arm']==arm and r['lr']==.001)
        baseline=next(r['accuracy'] for r in old['methods'] if r['method']=='ridge' and r['ridge']==.1)
        assert abs(rr[0]['before_accuracy']-baseline)<1e-7
        chosen=max([r for r in rr if r['mode']=='protected'],key=lambda r:(r['accuracy'],-(float('inf') if r['tau'] is None else r['tau'])))
        selection[arm]={'mode':'protected','tau':chosen['tau'],'validation_accuracy':chosen['accuracy'],'baseline_validation':baseline}
        dev.append({'arm':arm,'methods':rr,'baseline_reproduced':True});save('development.json',dev)
        print('DEV',arm,'base',round(100*baseline,2),'protected',round(100*chosen['accuracy'],2),'tau',chosen['tau'],flush=True)
    save('selection.json',{'per_arm':selection,'frozen_before_confirmation':True,
         'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()})
    records=json.loads((SOURCE/'results.json').read_text());rows=[]
    for seed in range(601,611):
        for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
            for arm in ARMS:
                state=torch.load(SOURCE/f'{seed}_{name}_{arm}.pt',weights_only=True)
                m=base.Model(seed,arm);m.load_state_dict(state['model']);mem=base.Memory()
                for k,v in state['memory'].items():setattr(mem,k,v)
                rr=evaluate(m,mem,tx,ty,order,all_settings())
                original=next(r for r in records if r['seed']==seed and r['order']==name and r['arm']==arm)
                old=next(r for r in original['methods'] if r['method']=='ridge')
                assert abs(rr[0]['before_accuracy']-old['accuracy'])<1e-7
                selected=next(r for r in rr if r['mode']=='protected' and r['tau']==selection[arm]['tau'])
                rows.append({'seed':seed,'order':name,'arm':arm,'baseline':old,'selected':selected,'methods':rr,
                    'historical_reference':original['diagnostics']['historical'],'direct':next(r for r in original['methods'] if r['method']=='direct'),
                    'baseline_reproduced':True,'model_and_memory_unchanged':True})
                save('results.json',rows)
                print('TEST',seed,name,arm,'base',round(100*old['accuracy'],2),'protected',round(100*selected['accuracy'],2),flush=True)
    save('completion.json',{'complete':len(rows)==60,'frozen_confirmation_models':len(rows),'development_reproductions':3,
                          'any_mean_above_direct':any(st.mean(r['selected']['accuracy'] for r in rows if r['arm']==arm)>.8553 for arm in ARMS)})


if __name__=='__main__':main()
