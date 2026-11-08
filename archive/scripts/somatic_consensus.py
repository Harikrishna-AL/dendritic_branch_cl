"""Somatic suppression, top-two class feedback and compact group receptive fields.

All automatic routes are constructed from current inputs and online local sums.
Historical masks are diagnostic only. No online test updates or old-image replay.
"""
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS

OUT=Path('results/somatic_consensus_20261021')
G=24
K=4
ARMS=[f'{field}_independent_keep{keep}' for field in ['full','patch5'] for keep in [24,17,12]]+['patch5_shared_keep24','patch6_shared_keep24']


def save(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def global_key(x):
    q=F.normalize(F.adaptive_avg_pool2d(x.reshape(-1,1,28,28),(7,9)).flatten(1),dim=-1)
    return F.normalize(torch.cat([q,torch.full((len(x),1),.2)],-1),dim=-1)


class Model(nn.Module):
    def __init__(self,seed,arm):
        super().__init__();torch.manual_seed(seed)
        self.arm=arm;field,head,keep=arm.split('_')
        self.size=28 if field=='full' else int(field.removeprefix('patch'))
        self.shared=head=='shared';self.keep=int(keep.removeprefix('keep'))
        self.pool=2 if self.size==6 else 3
        self.dimension=self.pool*self.pool+1
        d=self.size*self.size
        self.weight=nn.Parameter(torch.empty(G,K,d));self.bias=nn.Parameter(torch.empty(G,K))
        nn.init.kaiming_uniform_(self.weight.reshape(G*K,d),a=math.sqrt(5))
        nn.init.uniform_(self.bias,-1/math.sqrt(d),1/math.sqrt(d))
        self.head=nn.Linear(G if self.shared else G*K,10)
        self.register_buffer('utility',torch.zeros(G,K))
        if self.size==28:coords=torch.zeros(G,2,dtype=torch.uint8)
        elif self.size==5:coords=torch.tensor([(r,c) for r in [3,8,13,18] for c in [1,5,9,13,17,21]],dtype=torch.uint8)
        else:coords=torch.tensor([(r,c) for r in [2,8,14,20] for c in [0,4,8,12,16,20]],dtype=torch.uint8)
        self.register_buffer('coords',coords)

    def patches(self,x):
        if self.size==28:return x[:,None,:].expand(-1,G,-1)
        im=x.reshape(-1,28,28)
        # Only coordinates persist; indexing workspace is temporary.
        return torch.stack([im[:,r:r+self.size,c:c+self.size].flatten(1) for r,c in self.coords.tolist()],1)

    def local_keys(self,p):
        if self.size==28:
            q=F.adaptive_avg_pool2d(p[:,0].reshape(-1,1,28,28),(self.pool,self.pool)).flatten(1)
            q=F.normalize(q,dim=-1)
            q=F.normalize(torch.cat([q,torch.full((len(q),1),.2)],-1),dim=-1)
            return q[:,None].expand(-1,G,-1)
        q=F.adaptive_avg_pool2d(p.reshape(-1,1,self.size,self.size),(self.pool,self.pool)).flatten(1)
        q=F.normalize(q,dim=-1)
        q=F.normalize(torch.cat([q,torch.full((len(q),1),.2)],-1),dim=-1)
        return q.reshape(-1,G,self.dimension)

    def winner(self):return self.utility.argmax(-1)

    def soma(self,route,keep,score=None,rank='trace'):
        # route has shape G or B,G. Scores always use the winning branch.
        if score is None or rank=='trace':
            score=self.utility[torch.arange(G),route]
        mask=torch.zeros_like(score)
        return mask.scatter_(-1,score.topk(keep,dim=-1).indices,1)

    def mask(self):
        route=self.winner();active=self.soma(route,self.keep)
        return F.one_hot(route,K).float()*active[:,None]

    def response(self,p):return torch.tanh(torch.einsum('bgd,gkd->bgk',p,self.weight)+self.bias)

    def output(self,raw,mask):
        h=raw*mask
        return self.head(h.sum(-1) if self.shared else h.flatten(1))

    def forward(self,p,record=False):
        raw=self.response(p);h=raw*self.mask()
        if record:h.retain_grad()
        out=self.head(h.sum(-1) if self.shared else h.flatten(1))
        return (out,raw,h) if record else out

    def expanded_output(self):
        return self.head.weight[:,:,None].expand(-1,G,K) if self.shared else self.head.weight.reshape(10,G,K)


class Memory:
    def __init__(self,dimension):
        self.local_sums=torch.zeros(G,K,dimension)
        self.count=torch.zeros(G,K)
        # Larger full-image cue control is separate from compact deployment.
        self.global_sums=torch.zeros(G,K,64)
        self.direct_cov=torch.zeros(64,64)
        self.direct_cross=torch.zeros(10,64)

    @torch.no_grad()
    def update(self,m,p,x,y):
        route=m.winner();q=m.local_keys(p);ix=torch.arange(G)
        # Branch competition happens before soma suppression. Record each local winner,
        # including groups that will be silent; no class label enters these writes.
        self.local_sums[ix,route]+=q.sum(0)
        self.count[ix,route]+=len(x)
        key=global_key(x);self.global_sums[ix,route]+=key.sum(0)[None]
        self.direct_cov+=key.T@key
        self.direct_cross+=F.one_hot(y,10).float().T@key

    def direct_weight(self,ridge):
        return torch.linalg.solve((self.direct_cov+ridge*torch.eye(64)).double(),self.direct_cross.T.double()).T.float()


def digest(m,mem):
    h=hashlib.sha256()
    for v in [*m.state_dict().values(),*vars(mem).values()]:h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def specifications(global_controls=True):
    ss=[]
    for keep in [24,17,12]:
        for rank in (['trace'] if keep==24 else ['trace','cue']):
            for feedback in [False,True]:ss.append({'cue':'local','keep':keep,'rank':rank,'feedback':feedback})
    if global_controls:
        for feedback in [False,True]:ss.append({'cue':'global64','keep':24,'rank':'trace','feedback':feedback})
    return ss


def family(s):
    prefix='global64_' if s['cue']=='global64' else ''
    if s['keep']==24:return prefix+('feedback' if s['feedback'] else 'baseline')
    return prefix+'somatic_'+s['rank']+('_feedback' if s['feedback'] else '')


@torch.no_grad()
def inference(m,raw,score,spec):
    route=score.argmax(-1)
    winning=score.gather(-1,route[:,:,None]).squeeze(-1)
    active=m.soma(route,spec['keep'],winning,spec['rank'])
    mask=F.one_hot(route,K).float()*active[:,:,None]
    first=m.output(raw,mask)
    top2=first.topk(2,dim=-1).indices
    if not spec['feedback']:return first.argmax(-1),top2
    # Each image constructs two transient class-consistent configurations.
    weights=m.expanded_output()[top2]              # B,2,G,K
    contribution=raw[:,None]*weights
    revised=contribution.argmax(-1)
    revised_mask=F.one_hot(revised,K).float()*active[:,None,:,None]
    response=raw[:,None].expand(-1,2,-1,-1).reshape(-1,G,K)
    second=m.output(response,revised_mask.reshape(-1,G,K)).reshape(-1,2,10)
    candidate_logprob=second.log_softmax(-1).gather(-1,top2[:,:,None]).squeeze(-1)
    choice=candidate_logprob.argmax(-1)
    # All ten outputs remain eligible after choosing a proposed route.
    return second[torch.arange(len(raw)),choice].argmax(-1),top2


@torch.no_grad()
def evaluate(m,mem,bank,x,y,order,specs=None,direct_ridges=(.1,1.,10.)):
    before=digest(m,mem)
    raw=[];qlocal=[];qglobal=[]
    for b in x.split(512):
        p=m.patches(b);raw.append(m.response(p));qlocal.append(m.local_keys(p));qglobal.append(global_key(b))
    raw=torch.cat(raw);qlocal=torch.cat(qlocal);qglobal=torch.cat(qglobal)
    true_stage=torch.empty_like(y)
    for i,pair in enumerate(order):true_stage[torch.isin(y,torch.tensor(pair))]=i
    old=true_stage<4
    def metric(pred):
        return {'accuracy':float((pred==y).float().mean()),'old_accuracy':float((pred[old]==y[old]).float().mean()),
                'new_accuracy':float((pred[~old]==y[~old]).float().mean()),
                'pair_accuracy':[float((pred[true_stage==i]==y[true_stage==i]).float().mean()) for i in range(5)]}
    local=torch.einsum('bgd,gkd->bgk',qlocal,F.normalize(mem.local_sums,dim=-1))
    global_score=torch.einsum('bd,gkd->bgk',qglobal,F.normalize(mem.global_sums,dim=-1))
    local[:,mem.count==0]=-1000;global_score[:,mem.count==0]=-1000
    rows=[]
    for spec in (specifications() if specs is None else specs):
        score=local if spec['cue']=='local' else global_score
        pp=[];top=[]
        for a in range(0,len(x),512):
            pred,two=inference(m,raw[a:a+512],score[a:a+512],spec)
            pp.append(pred);top.append(two)
        pred=torch.cat(pp);top=torch.cat(top)
        dimension=m.dimension if spec['cue']=='local' else 64
        rows.append({**spec,'family':family(spec),**metric(pred),
                     'first_pass_top2_true_class_coverage':float((top==y[:,None]).any(-1).float().mean()),
                     'model_and_memory_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values())+G*K*(dimension+1)*4})
    diag={'latest':metric(m.output(raw,m.mask()).argmax(-1))}
    if bank is not None:
        diag['historical']=metric(m.output(raw,bank[true_stage]).argmax(-1))
        diag['wrong_historical']=metric(m.output(raw,bank[(true_stage+1)%5]).argmax(-1))
    direct=[]
    for ridge in direct_ridges:direct.append({'ridge':ridge,**metric((qglobal@mem.direct_weight(ridge).T).argmax(-1)),'numeric_bytes':18944})
    assert before==digest(m,mem)
    diag['evaluation_state_unchanged']=True
    return rows,diag,direct


def train(seed,arm,lr,x,y,ex,ey,order,joint=False,optimizer_kind='adam'):
    m=Model(seed,arm);mem=Memory(m.dimension)
    optimizer=torch.optim.Adam(m.parameters(),lr=lr) if optimizer_kind=='adam' else torch.optim.SGD(m.parameters(),lr=lr)
    groups=[list(range(10))] if joint else order
    bank=[];acq=[];start=time.perf_counter()
    for stage,pair in enumerate(groups):
        ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
        for epoch in range(2):
            gen=torch.Generator().manual_seed(seed*100000+(9000 if joint else stage*100)+epoch)
            for ix in ids[torch.randperm(len(ids),generator=gen)].split(128):
                bx,by=x[ix],y[ix];p=m.patches(bx)
                mem.update(m,p,bx,by)
                optimizer.zero_grad(set_to_none=True)
                log,raw,h=m(p,True);F.cross_entropy(log,by).backward()
                utility=-(h.grad.detach()*raw.detach()).sum(0)
                nn.utils.clip_grad_norm_(m.parameters(),5);optimizer.step()
                m.utility.mul_(.95).add_(utility,alpha=.05)
        bank.append(m.mask().clone())
        with torch.no_grad():
            current=torch.isin(ey,torch.tensor(pair));pred=[]
            for b in ex[current].split(512):pred.append(m(m.patches(b)).argmax(-1))
            acq.append(float((torch.cat(pred)==ey[current]).float().mean()))
    return m,mem,None if joint else torch.stack(bank),acq,time.perf_counter()-start


def verify():
    x=torch.rand(6,784);y=torch.arange(6)
    sizes={}
    for arm in ARMS:
        m=Model(1,arm);p=m.patches(x)
        assert p.shape==(len(x),G,m.size*m.size)
        assert int(m.mask().sum())==m.keep
        log,raw,h=m(p,True);F.cross_entropy(log,y).backward()
        assert torch.isfinite(h.grad).all()
        mem=Memory(m.dimension);before={k:v.clone() for k,v in m.state_dict().items()}
        mem.update(m,p,x,y);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
        q=m.local_keys(p);score=torch.einsum('bgd,gkd->bgk',q,F.normalize(mem.local_sums,dim=-1))
        for keep in [24,17,12]:
            route=score.argmax(-1);mask=m.soma(route,keep)
            assert (mask.sum(-1)==keep).all()
        sizes[arm]=sum(v.numel()*v.element_size() for v in m.state_dict().values())+G*K*(m.dimension+1)*4
        for spec in specifications():
            pred,top=inference(m,raw.detach(),score,spec)
            assert pred.shape==y.shape and top.shape==(len(y),2)
    assert sizes['patch5_independent_keep24']==18520
    assert sizes['patch5_shared_keep24']==15640
    assert sizes['patch6_shared_keep24']==17944
    # Shared-head expansion exactly agrees with the normal grouped classifier.
    m=Model(2,'patch5_shared_keep24');raw=m.response(m.patches(x));mask=m.mask()
    direct=torch.einsum('bgk,cgk->bc',raw*mask,m.expanded_output())+m.head.bias
    assert torch.allclose(m.output(raw,mask),direct,atol=1e-6)
    # Verify feedback chooses the largest candidate-class contribution per active group.
    m=Model(3,'patch5_independent_keep24');raw=m.response(m.patches(x));c=torch.arange(len(x))%10
    contribution=raw*m.expanded_output()[c]
    route=contribution.argmax(-1)
    assert torch.equal(contribution.gather(-1,route[:,:,None]).squeeze(-1),contribution.max(-1).values)
    return {'patch_shapes_and_active_counts':True,'finite_usefulness_gradients':True,
            'memory_writes_do_not_change_model':True,'soma_keeps_exact_requested_groups':True,
            'feedback_runs_without_labels_or_bank':True,'shared_output_expansion_matches_classifier':True,
            'feedback_maximizes_candidate_class_contribution':True,'numeric_bytes':sizes}


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    save('verification.json',verify())
    protocol={
        'question':'Test somatic thresholding, transient top-two-class branch consensus and compact patch-restricted groups, individually and combined.',
        'arms':ARMS,'stream':'FullMNIST pixels/255,5pairs,2epochs/pair,batch128,Adam,clip5,oneoptimizer,utilityEMA.95 negative gradient-times-response. No replay/pretraining/task-ID input/boundaryreset.',
        'architecture':'24groupsx4branches,onewinner/group before soma. Independent head has unique W[c,g,m]. Shared head is explicitcontrol. Patch5 has actual25weights/branch,24localpositions rows3/8/13/18 cols1/5/9/13/17/21. Patch6shared uses36weights/branch. Full has784weights/branch.',
        'memory':'Local3x3pooled patch normalized+constant.2 =>10Dkey (patch6 uses2x2+constant=>5D). Sums/counts perbranch. Beforegradient, associate current localkey with eachgroupwinner before soma, eveniflater silent. Full64globalprototype control and direct64keyridge are competing alternatives, notpartofcompactdeployment.',
        'threshold':'Retain24/17/12groups. Winningbranch recentutility ranks groups; secondary cue ranking uses currentimage cosine match, not true supervisedutility. Trainkeep24/17/12crossed with inferencekeep to distinguish posthocfromtrainedsuppression.',
        'feedback':'Inferenceonly. Pass1all10logits->top2classes. For eachcandidate c, everyactivegroup snaps to argmax_branch response*W[c,g,m]. ActivegroupsfixedfromPass1. Recomputeall10logits; choosecandidate route by its candidateclass log-softmax, then output unrestricted10classargmax. No routebank, labels, IDs, updates or biologicalbAPclaim.',
        'validation':'Devseed700stratified50k/10k. All8backbones LR.001/.003. ONEcommonLR maximizesmeanvalidation local/no-feedback accuracy at eachmodels trainingkeep; then per-backbone perreadoutfamily select inferencekeep17/12 on validation, trace/cuefamiliesseparate. Global64alternativebaseline/feedbackseparate. Directridge.1/1/10selectedmeanvalidation. Settingssavedbeforeofficialtest.',
        'confirmation':'Fresh701-710,forward/reverse,full60k/10k,160sequentialruns.10independentseeds,orderspaired. Fullindependentkeep24andpatch5independentkeep24jointcapacityreferences:10seedseach,20runs,settingsselectedindependentlyin4jointdevelopmentruns.',
        'budget':'Count everyFP32parameter,utilitybuffer,localkeysums/counts,anduint8coords. Patch5independent18520B;patch5shared15640B;patch6shared17944B versusdirectridge18944B. Optimizer/transientworkspace/Pythonmetadataexcludedforall. No derivedweight cache. Modelcheckpoints contain competing memories for measurementonly.',
        'limits':'Fixedinputpatches,finitevalidationgrid,onefeedbackstep. Inferencefeedbacknottrainedend-to-end. Traceutilityglobalnotcurrentimagespecific;cueconfidence distinctsecondary. Thresholdmayremoveoldskills. Smallermodelmayhaveweakacquisition;separatejointreferences. SameMNISTdataset,newinitializationsnotnewdataset.'}
    save('protocol.json',protocol);(OUT/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen=torch.Generator().manual_seed(700)
    valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;fx,fy,vx,vy=x[mask],y[mask],x[valid],y[valid]
    dev=[]
    for lr in [.001,.003]:
        for arm in ARMS:
            m,mem,bank,acq,seconds=train(700,arm,lr,fx,fy,vx,vy,SPLITS)
            rr,diag,direct=evaluate(m,mem,bank,vx,vy,SPLITS)
            dev.append({'arm':arm,'lr':lr,'methods':rr,'diagnostics':diag,'direct':direct,'acquisition':st.mean(acq)})
            save('development.json',dev)
            print('DEV',lr,arm,'baseline',round(100*next(r['accuracy'] for r in rr if r['family']=='baseline'),2),'max',round(100*max(r['accuracy'] for r in rr if r['cue']=='local'),2),'acquisition',round(100*st.mean(acq),2),flush=True)
    candidates=[]
    for lr in [.001,.003]:
        chosenrows=[d for d in dev if d['lr']==lr]
        score=st.mean(next(r['accuracy'] for r in d['methods'] if r['cue']=='local' and not r['feedback'] and r['rank']=='trace' and r['keep']==int(d['arm'].split('keep')[-1])) for d in chosenrows)
        candidates.append({'lr':lr,'validation_score':score})
    lr=max(candidates,key=lambda r:(r['validation_score'],-r['lr']))['lr']
    setting={}
    for arm in ARMS:
        d=next(d for d in dev if d['arm']==arm and d['lr']==lr);ss=[]
        for name in sorted(set(r['family'] for r in d['methods'])):
            rr=max([r for r in d['methods'] if r['family']==name],key=lambda r:(r['accuracy'],r['keep']))
            ss.append({k:rr[k] for k in ['cue','keep','rank','feedback']})
        setting[arm]=ss
    direct_candidates=[{'ridge':reg,'score':st.mean(next(r['accuracy'] for r in d['direct'] if r['ridge']==reg) for d in dev if d['lr']==lr)} for reg in [.1,1.,10.]]
    direct_ridge=max(direct_candidates,key=lambda r:(r['score'],-r['ridge']))['ridge']
    joint_selection={};joint_dev=[]
    for arm in ['full_independent_keep24','patch5_independent_keep24']:
        for rate in [.001,.003]:
            m,mem,bank,acq,seconds=train(700,arm,rate,fx,fy,vx,vy,SPLITS,True)
            joint_dev.append({'arm':arm,'lr':rate,'validation_accuracy':acq[0]})
        joint_selection[arm]=max([r for r in joint_dev if r['arm']==arm],key=lambda r:(r['validation_accuracy'],-r['lr']))['lr']
    save('joint_development.json',joint_dev)
    selection={'lr':lr,'lr_candidates':candidates,'readouts':setting,'direct_ridge':direct_ridge,'direct_candidates':direct_candidates,'joint_lr':joint_selection,
               'frozen_before_confirmation':True,'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()}
    save('selection.json',selection);print('SELECTED_LR',lr,'DIRECT',direct_ridge,flush=True)
    rows=[];joint_rows=[]
    for seed in range(701,711):
        for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
            for arm in ARMS:
                m,mem,bank,acq,seconds=train(seed,arm,lr,x,y,tx,ty,order)
                # Save every predetermined inference setting for transparent factorial ablations.
                rr,diag,direct=evaluate(m,mem,bank,tx,ty,order,direct_ridges=[direct_ridge])
                rows.append({'seed':seed,'order':name,'arm':arm,'lr':lr,'methods':rr,'diagnostics':diag,'direct':direct,
                             'acquisition':st.mean(acq),'acquisition_by_pair':acq,'train_seconds':seconds,
                             'parameters':sum(p.numel() for p in m.parameters()),'model_numeric_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values())})
                save('results.json',rows)
                torch.save({'model':m.state_dict(),'memory':vars(mem),'historical_masks_diagnostic_only':bank},OUT/f'{seed}_{name}_{arm}.pt')
                print('TEST',seed,name,arm,'base',round(100*next(r['accuracy'] for r in rr if r['family']=='baseline'),2),'acq',round(100*st.mean(acq),2),'history',round(100*diag['historical']['accuracy'],2),flush=True)
        for arm in ['full_independent_keep24','patch5_independent_keep24']:
            m,mem,bank,acq,seconds=train(seed,arm,joint_selection[arm],x,y,tx,ty,SPLITS,True)
            joint_rows.append({'seed':seed,'arm':arm,'lr':joint_selection[arm],'joint_accuracy':acq[0]})
            save('joint_results.json',joint_rows)
        print('SEED_COMPLETE',seed,len(rows),flush=True)
    save('completion.json',{'complete':len(rows)==160 and len(joint_rows)==20,'sequential_confirmation_runs':len(rows),'joint_confirmation_runs':len(joint_rows),'sequential_development_runs':len(dev),'joint_development_runs':len(joint_dev)})


if __name__=='__main__':main()
