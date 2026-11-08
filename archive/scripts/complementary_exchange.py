"""Balanced complementary/redundant receptive fields x sensory communication.

One global validation-selected learning rate and ridge strength for all cells.
No replay, task-ID routing, historical-bank automatic readout, or test updates.
"""
import copy
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw, SPLITS

OUT=Path('results/complementary_exchange_20261021')
STRUCTURES=['grouped_shared','global_shared','global_independent']
ARMS=[f'{s}__{v}__{c}' for s in STRUCTURES
      for v in ['complementary','redundant'] for c in ['self','neighbor']]
N=24
U=N*4


def save(name,value):
    target=OUT/name
    temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(target)


def keys(x):
    q=F.normalize(F.adaptive_avg_pool2d(x.reshape(-1,1,28,28),(7,9)).flatten(1),dim=-1)
    return F.normalize(torch.cat([q,torch.full((len(x),1),.2)],-1),dim=-1)


def patches(x):
    im=x.reshape(-1,28,28)
    return torch.stack([im[:,r:r+14,c:c+14].flatten(1)
                        for r,c in [(0,0),(0,14),(14,0),(14,14)]],1)


def local_keys(x):
    im=x.reshape(-1,28,28)
    result=[]
    for r,c in [(0,0),(0,14),(14,0),(14,14)]:
        isolated=torch.zeros_like(im)
        isolated[:,r:r+14,c:c+14]=im[:,r:r+14,c:c+14]
        result.append(keys(isolated.flatten(1)))
    return torch.stack(result,1)


class Model(nn.Module):
    def __init__(self,seed,arm):
        super().__init__()
        self.structure,self.view,self.communication=arm.split('__')
        self.arm=arm
        torch.manual_seed(seed)
        self.weight=nn.Parameter(torch.empty(U,196))
        self.bias=nn.Parameter(torch.empty(U))
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        nn.init.uniform_(self.bias,-1/math.sqrt(196),1/math.sqrt(196))
        self.gain=nn.Parameter(torch.full((N,4),.1))
        self.shared=self.structure!='global_independent'
        self.head=nn.Linear(N if self.shared else U,10)
        self.register_buffer('utility',torch.zeros(U))
        group=torch.arange(N)[:,None]
        field=(group+torch.arange(4)[None])%4 if self.view=='complementary' else (group%4).expand(N,4)
        self.register_buffer('field_id',field.flatten().clone())
        perm=self.field_id.argsort(stable=True)
        self.register_buffer('field_permutation',perm)
        self.register_buffer('field_inverse',perm.argsort())
        # In complementary groups, cycle around geometrically adjacent quarters.
        cycle=torch.tensor([0,1,3,2])
        neighbor=[]
        for g in range(N):
            if self.view=='complementary':
                where=(cycle[None,:]==field[g,:,None]).long().argmax(-1)
                wanted=torch.stack([cycle[(where-1)%4],cycle[(where+1)%4]],-1)
                ids=(field[g,None,:,None]==wanted[:,None,:]).long().argmax(1)
            else:
                k=torch.arange(4);ids=torch.stack([(k-1)%4,(k+1)%4],-1)
            neighbor.append(ids)
        self.register_buffer('neighbors',torch.stack(neighbor))

    def select(self,score):
        if self.structure=='grouped_shared':
            score=score.reshape(*score.shape[:-1],N,4)
            return torch.zeros_like(score).scatter_(-1,score.argmax(-1,keepdim=True),1).flatten(-2)
        return torch.zeros_like(score).scatter_(-1,score.topk(N,dim=-1).indices,1)

    def config(self):return self.select(self.utility)

    def preactivation(self,x):
        p=patches(x)
        w=self.weight[self.field_permutation].reshape(4,U//4,196)
        z=torch.einsum('bfd,fnd->bfn',p,w).flatten(1)[:,self.field_inverse]
        return z+self.bias

    def response(self,x,mode=None):
        z=self.preactivation(x).reshape(-1,N,4)
        evidence=torch.tanh(z)
        mode=self.communication if mode is None else mode
        if mode=='neighbor':
            incoming=evidence[:,torch.arange(N)[:,None,None],self.neighbors].mean(-1)
        else:incoming=evidence
        return (.75*evidence+.25*torch.tanh(z+self.gain*incoming)).flatten(1)

    def classify(self,response,config):
        h=response*config
        if self.shared:h=h.reshape(-1,N,4).sum(-1)
        return self.head(h)

    def forward(self,x,record=False):
        raw=self.response(x);h=raw*self.config()
        if record:h.retain_grad()
        log=self.head(h.reshape(-1,N,4).sum(-1) if self.shared else h)
        return (log,raw,h) if record else log


class Memory:
    def __init__(self):
        self.cov=torch.zeros(64,64)
        self.mask_cross=torch.zeros(U,64)
        self.class_cross=torch.zeros(10,64)
        self.local_sums=torch.zeros(U,64)

    @torch.no_grad()
    def update(self,x,y,config,field_id):
        q=keys(x)
        self.cov+=q.T@q
        self.mask_cross+=config[:,None]*q.sum(0)[None]
        self.class_cross+=F.one_hot(y,10).float().T@q
        regional=local_keys(x).sum(0)
        self.local_sums+=config[:,None]*regional[field_id]

    def coefficients(self,cross,ridge):
        return torch.linalg.solve((self.cov+ridge*torch.eye(64)).double(),cross.T.double()).T.float()


def digest(m,mem):
    h=hashlib.sha256()
    for v in [*m.state_dict().values(),*vars(mem).values()]:h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(m,mem,bank,x,y,order,ridges=(.1,1.,10.),direct_ridges=(.1,1.,10.),message_mode=None):
    before=digest(m,mem)
    raw=torch.cat([m.response(b,mode=message_mode) for b in x.split(512)])
    q=torch.cat([keys(b) for b in x.split(512)])
    region=torch.cat([local_keys(b) for b in x.split(512)])
    true_stage=torch.empty_like(y)
    for i,p in enumerate(order):true_stage[torch.isin(y,torch.tensor(p))]=i
    old=true_stage<4
    def metric(pred):
        return {'accuracy':float((pred==y).float().mean()),
                'old_accuracy':float((pred[old]==y[old]).float().mean()),
                'new_accuracy':float((pred[~old]==y[~old]).float().mean()),
                'pair_accuracy':[float((pred[true_stage==i]==y[true_stage==i]).float().mean()) for i in range(5)]}
    def classify(c):return m.classify(raw,c).argmax(-1)
    diagnostics={'latest':metric(classify(m.config())),
                 'historical':metric(classify(bank[true_stage])),
                 'wrong_historical':metric(classify(bank[(true_stage+1)%5]))}
    rows=[]
    for ridge in ridges:
        c=m.select(q@mem.coefficients(mem.mask_cross,ridge).T)
        expected=bank[true_stage]
        rows.append({'method':'ridge','ridge':ridge,**metric(classify(c)),
                     'historical_active_overlap':float(((c*expected).sum(-1)/N).mean()),
                     'whole_config_match':float((c==expected).all(-1).float().mean())})
    local=torch.einsum('bud,ud->bu',region[:,m.field_id],F.normalize(mem.local_sums,dim=-1))
    local[:,mem.local_sums.norm(dim=-1)==0]=-1000
    rows.append({'method':'local_field',**metric(classify(m.select(local)))})
    for ridge in direct_ridges:
        rows.append({'method':'direct','ridge':ridge,**metric((q@mem.coefficients(mem.class_cross,ridge).T).argmax(-1))})
    assert before==digest(m,mem)
    diagnostics['evaluation_state_unchanged']=True
    return rows,diagnostics


def train(seed,arm,lr,x,y,ex,ey,order):
    m=Model(seed,arm);mem=Memory()
    optimizer=torch.optim.Adam(m.parameters(),lr=lr)
    bank=[];acquisition=[];start=time.perf_counter()
    for stage,pair in enumerate(order):
        ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
        for epoch in range(2):
            gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ix in ids[torch.randperm(len(ids),generator=gen)].split(128):
                bx,by=x[ix],y[ix]
                mem.update(bx,by,m.config(),m.field_id)
                optimizer.zero_grad(set_to_none=True)
                logits,raw,h=m(bx,True)
                loss=F.cross_entropy(logits,by);loss.backward()
                utility=-(h.grad.detach()*raw.detach()).sum(0)
                nn.utils.clip_grad_norm_(m.parameters(),5);optimizer.step()
                m.utility.mul_(.95).add_(utility,alpha=.05)
        bank.append(m.config().clone())
        with torch.no_grad():
            current=torch.isin(ey,torch.tensor(pair))
            pred=torch.cat([m(b).argmax(-1) for b in ex[current].split(512)])
            acquisition.append(float((pred==ey[current]).float().mean()))
    return m,mem,torch.stack(bank),acquisition,time.perf_counter()-start


def verify():
    torch.manual_seed(5);x=torch.rand(5,784);y=torch.arange(5)
    counts={}
    for arm in ARMS:
        m=Model(4,arm)
        assert torch.equal(torch.bincount(m.field_id),torch.full((4,),24))
        if m.view=='complementary':assert all(len(set(row.tolist()))==4 for row in m.field_id.reshape(N,4))
        else:assert (m.field_id.reshape(N,4)==m.field_id.reshape(N,4)[:,:1]).all()
        direct=(patches(x)[:,m.field_id]*m.weight[None]).sum(-1)+m.bias
        assert torch.allclose(m.preactivation(x),direct,atol=1e-6)
        assert int(m.config().sum())==24
        if m.structure=='grouped_shared':assert (m.config().reshape(N,4).sum(-1)==1).all()
        counts[arm]=sum(p.numel() for p in m.parameters())
        log,raw,h=m(x,True);F.cross_entropy(log,y).backward()
        assert m.gain.grad is not None and torch.isfinite(m.gain.grad).all()
        with torch.no_grad():
            m.gain.zero_()
            assert torch.equal(m.response(x,'self'),m.response(x,'neighbor'))
    for view in ['complementary','redundant']:
        for comm in ['self','neighbor']:
            assert counts[f'grouped_shared__{view}__{comm}']==counts[f'global_shared__{view}__{comm}']
    # Outside-field independence holds before exchange, with actual reduced input matrices.
    m=Model(2,'grouped_shared__complementary__neighbor')
    z=m.preactivation(x)
    for field in range(4):
        p=patches(x).clone();p[:,(field+1)%4]+=.5
        im=torch.empty(len(x),28,28)
        for k,(r,c) in enumerate([(0,0),(0,14),(14,0),(14,14)]):im[:,r:r+14,c:c+14]=p[:,k].reshape(-1,14,14)
        assert torch.equal(z[:,m.field_id==field],m.preactivation(im.flatten(1))[:,m.field_id==field])
    # Derivative underlying usefulness scores.
    log,raw,h=m(x,True);loss=F.cross_entropy(log,y);loss.backward()
    d=torch.randn_like(h);eps=.001
    def after_h(v):return F.cross_entropy(m.head(v.reshape(-1,N,4).sum(-1)),y)
    fd=(after_h(h.detach()+eps*d)-after_h(h.detach()-eps*d))/(2*eps)
    assert torch.allclose(fd,(h.grad*d).sum(),atol=.0003,rtol=.03)
    mem=Memory();initial=copy.deepcopy(m.state_dict());c=m.config();mem.update(x,y,c,m.field_id)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items())
    assert torch.allclose(mem.mask_cross,c[:,None]*keys(x).sum(0)[None])
    assert torch.allclose(mem.coefficients(mem.class_cross,1.),torch.linalg.solve(torch.eye(64).double()+keys(x).double().T@keys(x).double(),keys(x).double().T@F.one_hot(y,10).double()).T.float(),atol=1e-6)
    return {'balanced_pixel_coverage':True,'views_are_complementary_or_redundant_as_specified':True,
            'compact_projection_matches_direct':True,'outside_field_invariance_before_exchange':True,
            'zero_gain_message_equivalence':True,'active_count_24_all_models':True,
            'matched_tied_output_parameter_counts':True,'finite_exchange_gradients':True,
            'utility_derivative_finite_difference':True,'memory_writes_do_not_change_model':True,
            'preupdate_mask_targets':True,'online_ridge_matches_batch_reference':True,'parameter_counts':counts}


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    save('verification.json',verify())
    protocol={
        'question':'Does complementary sensory information cause a larger communication benefit than redundant sensory views, and does grouping supply a benefit over matched ordinary selected responses?',
        'arms':ARMS,'primary':'Automatic ten-digit accuracy using online image-to-mask ridge retrieval. Primary interaction=(neighbor-self)complementary-(neighbor-self)redundant, separately within each structure.',
        'secondary':'Old/new automatic accuracy, historical config oracle, acquisition, field-only associative readout, direct classifier baseline. All comparisons and failed cells retained.',
        'architecture':'24groups x4responses=96, exactly24active. Each response has196actual input weights, not masked784 tensors. Complementary groups see four different quarters; redundant groups see one quarter, balanced6groups/quarter. Both conditions have24responses/quarter globally.',
        'controls':'Grouped shared outputs: choose1/4,24-column head. Global shared: choose24/96, same tied output matrix and exact parameter count. Global independent: choose24/96,96-column head;720extra parameters, openly reported.',
        'communication':'One direct sensory exchange step. Evidence e=tanh(z), response=.75e+.25tanh(z+gain*incoming),96learnedgains initialized.1. Neighborincoming averages2neighbors within group; geometryTL-TR-BR-BL for complementary groups. Self uses own e, same parameters/depth. Communication is before selection; gradients can reach inactive senders.',
        'training':'MNISTpixels/255,2epochs/pair, batch128, Adam, clip5, utilityEMA.95 of negative gradient-times-response. No replay, pretraining, task-ID input, optimizer/trace boundaryreset, usageprotection.',
        'associations':'Current preupdate mask to fixed64wholeimagekeys. Covariance/crossstatistics only; no examples or historicalmaskbank. Labels supervise backbone and directclassifier, not maskmemory. Field-only prototype sums additionally stored for separate limited-view readout.',
        'selection':'Freshdevelopmentseed600,stratified50k/10k. Train all12cells at LR.001/.003. Choose ONE common LR/ridge(.1/1/10) by mean automaticvalidation acrossall12cells, with lowerLR/ridge tiebreak. Directridge separately by meanvalidation. Primary settings frozen before officialtest. This clean common-setting factorial may not maximize each individual architecture.',
        'confirmation':'Freshseeds601-610, forward/reverseorders,full60k/10k.240confirmationruns,10independentinitializations, orderspaired.24developmentruns. No confirmation retuning or optional stopping.',
        'statistics':'Paired order-averaged seed differences;95%tintervaldf9; primarydifference-of-differences perstructure; secondary comparisons exploratory/multiple, no universal/datasetgeneralizationclaim.',
        'state':'Evaluation checksums assert model andmemoryunchanged. Historicalmaskoracle uses truepair; automatic predictions never access bank or testlabels. Oldweightsneverrestored. Checkpoints are artifacts only.',
        'limits':'SameMNISTdataset as prior studies, fresh initializations not fresh datasets. Directclassifier stronger baseline. Redundant fields still have distinct learned filters. Globally balanced imagecoverage does not imply identical pergroupinformation. No claim of biologicallocality.'}
    save('protocol.json',protocol)
    (OUT/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen=torch.Generator().manual_seed(600)
    valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;fx,fy,vx,vy=x[mask],y[mask],x[valid],y[valid]
    dev=[]
    for lr in [.001,.003]:
        for arm in ARMS:
            m,mem,bank,acq,seconds=train(600,arm,lr,fx,fy,vx,vy,SPLITS)
            rr,diag=evaluate(m,mem,bank,vx,vy,SPLITS)
            dev.append({'arm':arm,'lr':lr,'methods':rr,'diagnostics':diag,'acquisition':st.mean(acq),'seconds':seconds})
            save('development.json',dev)
            print('DEV',lr,arm,'automatic',round(100*max(r['accuracy'] for r in rr if r['method']=='ridge'),2),'historical',round(100*diag['historical']['accuracy'],2),flush=True)
    candidates=[]
    for lr in [.001,.003]:
        for ridge in [.1,1.,10.]:
            score=st.mean(next(r['accuracy'] for r in d['methods'] if r['method']=='ridge' and r['ridge']==ridge) for d in dev if d['lr']==lr)
            candidates.append({'lr':lr,'ridge':ridge,'mean_validation_accuracy':score})
    chosen=max(candidates,key=lambda d:(d['mean_validation_accuracy'],-d['lr'],-d['ridge']))
    dc=[]
    for ridge in [.1,1.,10.]:
        dc.append({'ridge':ridge,'mean_validation_accuracy':st.mean(next(r['accuracy'] for r in d['methods'] if r['method']=='direct' and r['ridge']==ridge) for d in dev if d['lr']==chosen['lr'])})
    direct=max(dc,key=lambda d:(d['mean_validation_accuracy'],-d['ridge']))
    selection={'chosen':chosen,'candidates':candidates,'direct_chosen':direct,'direct_candidates':dc,
               'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
               'frozen_before_confirmation':True}
    save('selection.json',selection)
    print('SELECTED',selection,flush=True)
    rows=[]
    for seed in range(601,611):
        for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
            for arm in ARMS:
                m,mem,bank,acq,seconds=train(seed,arm,chosen['lr'],x,y,tx,ty,order)
                rr,diag=evaluate(m,mem,bank,tx,ty,order,[chosen['ridge']],[direct['ridge']])
                rows.append({'seed':seed,'order':name,'arm':arm,'lr':chosen['lr'],'methods':rr,
                             'diagnostics':diag,'acquisition':st.mean(acq),'acquisition_by_pair':acq,
                             'train_and_memory_seconds':seconds,'parameters':sum(p.numel() for p in m.parameters()),
                             'model_numeric_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values())})
                save('results.json',rows)
                torch.save({'model':m.state_dict(),'memory':vars(mem),'historical_bank_diagnostic_only':bank},OUT/f'{seed}_{name}_{arm}.pt')
                auto=next(r for r in rr if r['method']=='ridge')
                print('TEST',seed,name,arm,'automatic',round(100*auto['accuracy'],2),'old',round(100*auto['old_accuracy'],2),'historical',round(100*diag['historical']['accuracy'],2),flush=True)
        print('SEED_COMPLETE',seed,len(rows),flush=True)
    save('completion.json',{'complete':len(rows)==240,'confirmation_runs':len(rows),'development_runs':len(dev),
                            'seeds':list(range(601,611)),'orders':['forward','reverse']})


if __name__=='__main__':main()
