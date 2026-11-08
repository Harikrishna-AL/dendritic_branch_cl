"""Compact independent cue-group statistics, compared to full covariance."""
import argparse
import json
import statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.slow_route_consolidation import Memory,run,evaluate,diagnostics


class BlockMemory(Memory):
    def __init__(self,width,seed,filtered,block):
        super().__init__(width,seed,filtered)
        self.block=block;self.nblocks=width//block
        self.gram=torch.zeros(self.nblocks,block+1,block+1,dtype=torch.float64)
        self.cross=torch.zeros(self.nblocks,block+1,106,dtype=torch.float64)

    @torch.no_grad()
    def observe(self,q,y,route,reliable,target=None):
        ids=reliable if self.filtered else torch.ones(len(q),dtype=torch.bool)
        z=self.features(q[ids]);yy=y[ids]
        if not len(z):return
        tag=F.one_hot(route,4).flatten().float().expand(len(q),96) if target is None else target.flatten(1)
        tag=tag[ids]
        targets=torch.cat([tag,F.one_hot(yy,10).float()],1).double()
        grouped=z[:,:-1].reshape(len(z),self.nblocks,self.block).transpose(0,1).double()
        grouped=torch.cat([grouped,torch.ones(self.nblocks,len(z),1,dtype=torch.float64)],-1)
        self.gram.add_(grouped.transpose(1,2)@grouped)
        self.cross.add_(grouped.transpose(1,2)@targets)
        choices=tag.reshape(-1,24,4).argmax(-1);coactivity=F.one_hot(choices,4).double()
        self.links.add_(torch.einsum('bni,bnj->nij',coactivity,torch.roll(coactivity,-1,1)))
        self.accepted+=len(z)

    def solve(self,ridge):
        penalty=torch.eye(self.block+1,dtype=torch.float64)*ridge;penalty[-1,-1]=0
        grouped=torch.linalg.solve(self.gram+penalty,self.cross)/self.nblocks
        head=torch.cat([grouped[:,:-1].reshape(self.width,106),grouped[:,-1].sum(0,keepdim=True)],0)
        return head

    def bytes(self):
        return self.gram.numel()*8+self.nblocks*(self.block+1)*96*8+(self.width+1)*96*8+self.links.numel()*8+self.projection.numel()*4


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'development_seed':120,'confirmation_seeds':[121,122,123],'blocks':[16,64,128,512],
              'ridges':[.1,1.,10.],'gammas':[0.,.1,.5],
              'scope':'Reduce global cue covariance to independent small cue-group Gram matrices. Still an analytic engineering rule, not biologically local branch plasticity. No archive/replay/taskIDs. Current teacher is batch-global usefulness route;512random cues inherited. Two current-task epochs.',
              'selection':'Fresh stratified50k/10k training development selects block/ridge/communication on validation; unfiltered writing inherited. Fresh full60k confirmation, both orders. Full-covariance baseline separately tuned on same validation. Compact model maintains ONLY block statistics during stream; full baseline trained separately for paired audit.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=torch.Generator().manual_seed(120)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
    fit=torch.ones(len(y),dtype=torch.bool);fit[val]=False;candidates=[]
    for block in protocol['blocks']:
        factory=lambda w,s,f:BlockMemory(w,s,f,block)
        model,memories,_,_=run(120,x,y,fit,list(SPLITS),[512],memory_factory=factory)
        memory=memories[512,False]
        for ridge in protocol['ridges']:
            head=memory.solve(ridge)
            for gamma in protocol['gammas']:
                result=evaluate(model,memory,x[val],y[val],head,gamma);result.pop('routes')
                candidates.append({'block':block,'ridge':ridge,'gamma':gamma,**result,'numeric_bytes':memory.bytes()})
        print('dev block',block,round(max(r['accuracy'] for r in candidates if r['block']==block)*100,2),flush=True)
    compact=max([r for r in candidates if r['block']<512],key=lambda r:r['accuracy'])
    full=max([r for r in candidates if r['block']==512],key=lambda r:r['accuracy'])
    (out/'selection.json').write_text(json.dumps({'candidates':candidates,'compact':compact,'full':full},indent=2))
    print('selected compact',compact,'full',full,flush=True)
    rows=[]
    for order in ['forward','reverse']:
        pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
        for seed in protocol['confirmation_seeds']:
            reference=None
            for arm,selected in [('compact',compact),('full',full)]:
                factory=lambda w,s,f:BlockMemory(w,s,f,selected['block'])
                model,memories,history,seconds=run(seed,x,y,torch.ones(len(y),dtype=torch.bool),pairs,[512],memory_factory=factory)
                if reference is None:reference={k:v.clone() for k,v in model.state_dict().items()}
                else:assert all(torch.equal(v,reference[k]) for k,v in model.state_dict().items())
                memory=memories[512,False];head=memory.solve(selected['ridge'])
                result=evaluate(model,memory,tx,ty,head,selected['gamma'])
                row={'seed':seed,'order':order,'arm':arm,**diagnostics(model,tx,ty,result['routes'],history,pairs),'numeric_bytes':memory.bytes(),'seconds_all_controls':seconds,'accepted':memory.accepted}
                rows.append(row);print(order,seed,arm,round(row['accuracy']*100,2),flush=True)
                torch.save({'backbone':model.state_dict(),'projection':memory.projection,'head':head[:,:96].clone(),'gram':memory.gram,'cross':memory.cross[:,:,:96].clone(),'links':memory.links},out/f'{order}_{seed}_{arm}.pt')
                (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Compact cue-group routing statistics','','Analytic input-to-branch retrieval with independent groups of cue covariance. No example or configuration archive; extra statistical memory remains.','', '| Router | Final accuracy ± SD (six runs) | Numeric router memory |','|---|---:|---:|']
    for arm in ['compact','full']:
        rs=[r for r in rows if r['arm']==arm];a=[r['accuracy']*100 for r in rs]
        lines.append(f'| {arm} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% | {rs[0]["numeric_bytes"]:,} bytes |')
    lines+=['',f'Compact selection: `{compact}`.',f'Full selection: `{full}`.', '',
            'Both routers use identical final learned branch weights/classifiers, current-sample teacher routes and512random cue features. Compact protection learns each cue group independently and averages their branch-choice predictions; cross-group correlations are discarded. Numeric memory includes continuing-learning statistics, routing head, fixed cue projection and local neighbor links, excluding shared302,824-byte backbone, dataset buffers and temporary solve allocations. This is not a claim of strictly branch-local biological learning or unique dendritic advantage.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
