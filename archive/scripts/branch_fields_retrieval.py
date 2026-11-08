"""Prospective comparisons of grouping, output sharing, multiple branches and fields."""
import json
import math
import statistics as st
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw, SPLITS

OUT=Path('results/branch_fields_retrieval_20261021')
ARMS=['full_flat','full_group_independent','full_branch1','full_branch2',
      'quadrant_flat','quadrant_group_independent','quadrant_branch1','quadrant_branch2',
      'random_branch1']


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2))


def key(x):
    q=F.normalize(F.adaptive_avg_pool2d(x.reshape(-1,1,28,28),(7,9)).flatten(1),dim=-1)
    return F.normalize(torch.cat([q,torch.full((len(x),1),.2)],-1),dim=-1)


def fields(kind):
    if kind=='full':return torch.ones(4,784)
    masks=torch.zeros(4,28,28)
    if kind=='quadrant':
        for k in range(4):masks[k,(k//2)*14:(k//2+1)*14,(k%2)*14:(k%2+1)*14]=1
    else:
        gen=torch.Generator().manual_seed(774)
        ids=torch.randperm(784,generator=gen).reshape(4,196)
        masks=masks.flatten(1);masks.scatter_(1,ids,1)
    return masks.reshape(4,784)


class Model(nn.Module):
    def __init__(self,seed,arm):
        super().__init__();torch.manual_seed(seed)
        self.arm=arm;self.kind=arm.split('_')[0]
        self.grouped='flat' not in arm
        self.flat_count=50 if arm.endswith('flat50') else 25
        self.shared='branch' in arm
        self.participants=2 if arm.endswith('branch2') else 1
        self.hidden=nn.Linear(784,100)
        self.head=nn.Linear(25 if self.shared else 100,10)
        self.register_buffer('utility',torch.zeros(100))
        self.register_buffer('field',fields(self.kind))

    def select(self,score):
        if self.grouped:
            z=score.reshape(*score.shape[:-1],25,4)
            return torch.zeros_like(z).scatter_(-1,z.topk(self.participants,dim=-1).indices,1).flatten(-2)
        return torch.zeros_like(score).scatter_(-1,score.topk(self.flat_count,dim=-1).indices,1)

    def config(self):return self.select(self.utility)

    def response(self,x):
        return torch.tanh(F.linear(x,self.hidden.weight*self.field.repeat(25,1),self.hidden.bias))

    def classify(self,response,config):
        h=response*config
        if self.shared:h=h.reshape(-1,25,4).sum(-1)/math.sqrt(self.participants)
        return self.head(h)

    def forward(self,x,record=False):
        raw=self.response(x);h=raw*self.config()
        if record:h.retain_grad()
        pooled=h.reshape(-1,25,4).sum(-1)/math.sqrt(self.participants) if self.shared else h
        log=self.head(pooled)
        return (log,raw,h) if record else log


class Memory:
    def __init__(self,field):
        self.field=field.clone()
        self.cov=torch.zeros(64,64)
        self.cross=torch.zeros(100,64)
        self.direct=torch.zeros(10,64)
        self.sums={scope:torch.zeros(100,64) for scope in ['global','field']}
        self.links=torch.zeros(25,4,4)

    @torch.no_grad()
    def update(self,x,y,config):
        q=key(x);self.cov+=q.T@q
        self.cross+=config[:,None]*q.sum(0)[None]
        self.direct+=F.one_hot(y,10).float().T@q
        self.sums['global']+=config[:,None]*q.sum(0)[None]
        patch=torch.stack([key(x*f) for f in self.field],1)
        ps=patch.sum(0).repeat(25,1)
        self.sums['field']+=config[:,None]*ps
        c=config.reshape(25,4)
        self.links+=len(x)*c[:,:,None]*c.roll(-1,0)[:,None,:]

    def ridge(self,cross,lam):
        return torch.linalg.solve((self.cov+lam*torch.eye(64)).double(),cross.T.double()).T.float()


def checksum(m,mem):
    import hashlib
    h=hashlib.sha256()
    for v in m.state_dict().values():h.update(v.numpy().tobytes())
    for v in [mem.cov,mem.cross,mem.direct,mem.links,*mem.sums.values()]:h.update(v.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(m,mem,bank,x,y,order,settings):
    before=checksum(m,mem)
    raw=torch.cat([m.response(b) for b in x.split(512)])
    q=torch.cat([key(b) for b in x.split(512)])
    truth=torch.zeros(len(y),dtype=torch.long)
    for i,p in enumerate(order):truth[torch.isin(y,torch.tensor(p))]=i
    old=truth<4
    def metrics(pred):return {'accuracy':float((pred==y).float().mean()),'old_accuracy':float((pred[old]==y[old]).float().mean()),'new_accuracy':float((pred[~old]==y[~old]).float().mean())}
    def classify(c):return m.classify(raw,c).argmax(-1)
    latest=metrics(classify(m.config()))
    oracle=metrics(classify(bank[truth]))
    wrong=metrics(classify(bank[(truth+1)%5]))
    rows=[];patch=None;unaries={}
    for s in settings:
        method=s['method']
        if method=='direct':
            rows.append({**s,**metrics((q@mem.ridge(mem.direct,s['ridge']).T).argmax(-1))});continue
        if method=='ridge':score=q@mem.ridge(mem.cross,s['ridge']).T
        else:
            scope=s['scope']
            if scope not in unaries:
                if scope=='global':score=q@F.normalize(mem.sums[scope],dim=-1).T
                else:
                    if patch is None:patch=torch.stack([torch.cat([key(b*f) for b in x.split(512)]) for f in mem.field],1)
                    score=torch.einsum('bkd,nkd->bnk',patch,F.normalize(mem.sums[scope],dim=-1).reshape(25,4,64)).flatten(1)
                score[:,mem.sums[scope].norm(dim=-1)==0]=-1000
                unaries[scope]=score
            score=unaries[scope]
            gamma=s.get('gamma',0)
            if gamma:
                unary=score.reshape(-1,25,4)
                p=torch.softmax(unary/.05,-1)
                right=mem.links/mem.links.sum(-1,keepdim=True).clamp_min(1)
                left=mem.links/mem.links.sum(-2,keepdim=True).clamp_min(1)
                for _ in range(3):
                    msg=.5*(torch.einsum('bnl,nkl->bnk',p.roll(-1,1),right)+torch.einsum('bnk,nkl->bnl',p.roll(1,1),left.roll(1,0)))
                    p=torch.softmax((unary+gamma*msg)/.05,-1)
                score=p.flatten(1)
        c=m.select(score);pred=classify(c)
        active=bank[truth]
        overlap=(c*active).sum(-1)/active.sum(-1)
        rows.append({**s,**metrics(pred),'historical_active_overlap':float(overlap.mean()),
                     'whole_config_match':float((c==active).all(-1).float().mean())})
    assert checksum(m,mem)==before
    return rows,{'latest':latest,'historical':oracle,'wrong_historical':wrong,'evaluation_state_unchanged':True}


def settings(arm):
    rows=[{'method':'ridge','ridge':r} for r in [.1,1.,10.]]
    rows += [{'method':'direct','ridge':r} for r in [.1,1.,10.]]
    for scope in ['global','field']:
        for gamma in ([0.,.05,.2] if 'flat' not in arm else [0.]):
            rows.append({'method':'local','scope':scope,'gamma':gamma})
    return rows


def family(s):
    if s['method']!='local':return s['method']
    return 'local_'+s['scope']+('_comm' if s['gamma'] else '')


def train(seed,arm,lr,x,y,ex,ey,order,factory=None):
    m=(Model if factory is None else factory)(seed,arm);mem=Memory(m.field)
    opt=torch.optim.Adam(m.parameters(),lr=lr)
    bank=[];acq=[]
    for stage,pair in enumerate(order):
        ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
        for epoch in range(2):
            gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ix in ids[torch.randperm(len(ids),generator=gen)].split(128):
                bx,by=x[ix],y[ix]
                mem.update(bx,by,m.config())
                opt.zero_grad(set_to_none=True)
                log,raw,h=m(bx,True);loss=F.cross_entropy(log,by);loss.backward()
                utility=-(h.grad.detach()*raw.detach()).sum(0)
                nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                m.utility.mul_(.95).add_(utility,alpha=.05)
        bank.append(m.config().clone())
        with torch.no_grad():
            ids=torch.isin(ey,torch.tensor(pair))
            pred=torch.cat([m(b).argmax(-1) for b in ex[ids].split(512)])
            acq.append(float((pred==ey[ids]).float().mean()))
    return m,mem,torch.stack(bank),acq


def verify():
    for kind in ['quadrant','random']:
        f=fields(kind);assert torch.equal(f.sum(0),torch.ones(784));assert (f.sum(1)==196).all()
        m=Model(1,kind+'_branch1');x=torch.rand(3,784)
        a=m.response(x).reshape(-1,25,4)
        for k in range(4):
            changed=x+torch.rand_like(x)*(1-f[k])
            assert torch.equal(a[:,:,k],m.response(changed).reshape(-1,25,4)[:,:,k])
    m=Model(2,'full_branch2');x=torch.rand(5,784);log,raw,h=m(x,True)
    assert (m.config().reshape(25,4).sum(-1)==2).all()
    expected=m.head((raw*m.config()).reshape(-1,25,4).sum(-1)/math.sqrt(2))
    assert torch.equal(log,expected)
    y=torch.arange(5);F.cross_entropy(log,y).backward()
    assert torch.isfinite(h.grad).all()
    mem=Memory(m.field);c=m.config();mem.update(x,y,c)
    assert torch.allclose(mem.cross,c[:,None]*key(x).sum(0)[None])
    return {'field_partition_and_outside_field_invariance':True,'two_branch_pooling_and_gradients':True,'online_association_targets_current_mask':True}


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    save('verification.json',verify())
    save('protocol.json',{
        'question':'Do grouping, shared output weights, multi-branch participation or different receptive fields improve online input-driven configuration retrieval?',
        'arms':ARMS,'training':'MNIST pixels/255,2epochs/pair,batch128,Adam,clip5,EMA.95 negative gradient-times-response usefulness. No replay, pretraining, IDs, boundary resets, protection or bank in automatic readout.',
        'architecture':'100 tanh input responses. Flat selects25 globally; grouped independent selects1/4 with100-column head; branch1 selects1/4 with25-column shared head; branch2 selects2/4 and sums/sqrt2. Same hidden initialization and data perseed. Fixed receptive masks: full784, quadrants196, random196 pixel partition. Allocated dense tensors same but effective trainable input contacts differ.',
        'associations':'Current pre-update config associated with current fixed64pixelkeys. Ridge predicts100 mask bits from globalkeys. Local stores per-response key sums from global or corresponding field inputs. Cumulative writes, no labels to routing. Direct classifier gets same globalkeys and labels. All candidates sidecars coexist only for paired measurement.',
        'communication':'Three inference rounds of learned coactivation compatibility along a fixed ring of25groups. Fields meet through compatibility between alternatives in adjacent groups, not direct exchange of sensory activations between branches. gamma.05/.2 validationselected, off is same final weights/memory. Flat has no communication. No global inference-locality or biological claim.',
        'selection':'Development560 stratified50k/10k. Perarm LR.001/.003 chosen by historical recovery validation, acquisition tiebreak. Then per-readout ridge.1/1/10, gamma.05/.2 chosen by automaticvalidation. Local-off fixed. Test never selects.',
        'confirmation':'Fresh561-563, bothorders,full60k/10k. Threeindependentseeds,orderspaired.54confirmationbackbones. Oracle finalweightsfixed/truetaskconfig selection; automatic uses images only. Historical bank analysis only.',
        'limits':'Quadrant has fewer effective parameters; two branches has more active computation. No architecture advantage can be attributed to fields alone without this qualification. Ring is arbitrary, communication correlations are not necessarily useful information. MNISTsmall,3seeds.'})
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen=torch.Generator().manual_seed(560)
    valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
    fx,fy,vx,vy=x[mask],y[mask],x[valid],y[valid]
    selection={}
    for arm in ARMS:
        candidates=[];best=None
        for lr in [.001,.003]:
            m,mem,bank,acq=train(560,arm,lr,fx,fy,vx,vy,SPLITS)
            rr,diag=evaluate(m,mem,bank,vx,vy,SPLITS,settings(arm))
            candidates.append({'lr':lr,'methods':rr,'diagnostics':diag,'acquisition':st.mean(acq)})
            print('DEV',arm,lr,'historical',round(100*diag['historical']['accuracy'],2),'ridge',round(100*max(r['accuracy'] for r in rr if r['method']=='ridge'),2),flush=True)
        chosen=max(candidates,key=lambda r:(r['diagnostics']['historical']['accuracy'],r['acquisition']))
        selected=[]
        for name in sorted(set(family(s) for s in chosen['methods'])):
            s=max([s for s in chosen['methods'] if family(s)==name],key=lambda s:s['accuracy'])
            selected.append({k:s[k] for k in ['method','scope','gamma','ridge'] if k in s})
        selection[arm]={'lr':chosen['lr'],'readouts':selected,'candidates':candidates}
        save('selection.json',selection)
    rows=[]
    for seed in [561,562,563]:
        for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
            for arm in ARMS:
                chosen=selection[arm]
                m,mem,bank,acq=train(seed,arm,chosen['lr'],x,y,tx,ty,order)
                rr,diag=evaluate(m,mem,bank,tx,ty,order,chosen['readouts'])
                r={'seed':seed,'order':name,'arm':arm,'lr':chosen['lr'],'methods':rr,'diagnostics':diag,'acquisition':st.mean(acq),
                   'acquisition_by_pair':acq,'parameters_allocated':sum(p.numel() for p in m.parameters()),
                   'effective_parameters':sum(p.numel() for p in m.parameters())-(784-int(m.field[0].sum()))*100,
                   'model_numeric_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values())}
                rows.append(r);save('results.json',rows)
                torch.save({'model':m.state_dict(),'bank_analysis_only':bank,'memory':{'cov':mem.cov,'cross':mem.cross,'direct':mem.direct,'sums':mem.sums,'links':mem.links}},OUT/f'{seed}_{name}_{arm}.pt')
                print('TEST',seed,name,arm,'oracle',round(100*diag['historical']['accuracy'],2),[(family(s),round(100*s['accuracy'],2)) for s in rr],flush=True)
    save('completion.json',{'complete':len(rows)==54,'confirmation_backbones':len(rows),'development_backbones':18})


if __name__=='__main__':main()
