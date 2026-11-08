"""Fixed-budget image-context routing; no labels, task IDs or replay in router."""
import math
import torch
from torch import nn
from .bdh_wrapper import BDHWrapper


class OnlineRouter(nn.Module):
    def __init__(self, pixels=196, heads=4, dim=16, groups=3, seed=1234):
        super().__init__()
        self.heads, self.groups = heads, groups
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer('projection', torch.randn(heads, pixels, dim, generator=gen)/math.sqrt(pixels))
        for name, value in {
            'count':torch.zeros(heads,groups), 'total':torch.zeros(heads,groups,dim),
            'square':torch.zeros(heads,groups,dim), 'mean':torch.zeros(heads,groups,dim),
            'var':torch.ones(heads,groups,dim), 'threshold':torch.zeros(heads),
            'ready':torch.tensor(False),
        }.items(): self.register_buffer(name,value)

    def project(self, images):
        return torch.einsum('bp,hpd->bhd',(images.flatten(1).long() >> 4).float()/15.,self.projection)

    def scores(self,z,mode):
        delta=z[:,:,None,:]-self.mean[None]
        if mode=='centroid': return -delta.square().sum(-1)
        # Uniform component priors avoid historical frequency starving new groups.
        return -.5*(delta.square()/self.var[None]+self.var[None].log()).sum(-1)

    @torch.no_grad()
    def fit(self,images,mode):
        z=self.project(images)
        # Deterministic farthest-point initialization, followed by Lloyd iterations.
        centers=[z[0]]
        for _ in range(1,self.groups):
            dist=(z[:,:,None,:]-torch.stack(centers,1)[None]).square().sum(-1).amin(-1)
            ix=dist.argmax(0)
            centers.append(z[ix,torch.arange(self.heads,device=z.device)])
        self.mean.copy_(torch.stack(centers,1))
        for _ in range(10):
            labels=self.scores(z,'centroid').argmax(-1)
            r=torch.nn.functional.one_hot(labels,self.groups).float()
            n=r.sum(0).clamp_min(1)
            self.mean.copy_(torch.einsum('bhk,bhd->hkd',r,z)/n[:,:,None])
        self.count.copy_(r.sum(0))
        self.total.copy_(torch.einsum('bhk,bhd->hkd',r,z))
        self.square.copy_(torch.einsum('bhk,bhd->hkd',r,z.square()))
        self.refresh()
        # Fixed 99% training-coverage threshold, never calibrated using test/future data.
        self.threshold.copy_(torch.quantile(self.scores(z,mode).amax(-1),.01,dim=0))
        self.ready.fill_(True)

    def refresh(self):
        n=self.count.clamp_min(1)[:,:,None]
        self.mean.copy_(self.total/n)
        self.var.copy_((self.square/n-self.mean.square()).clamp_min(1e-4))

    @torch.no_grad()
    def route(self,images,mode):
        if not self.ready: raise RuntimeError('Router must be initialized from past training inputs')
        z=self.project(images)
        scores=self.scores(z,mode)
        best,choice=scores.max(-1)
        accepted=best>=self.threshold[None]
        gate=torch.zeros(len(images),self.heads,self.groups+1,device=images.device)
        gate[:,:,0]=1
        gate.scatter_(2,(choice+1)[:,:,None],accepted.float()[:,:,None])
        assignment=torch.where(accepted,choice+1,torch.zeros_like(choice))
        return gate,assignment,(z,scores,accepted)

    @torch.no_grad()
    def update(self,cache,mode):
        z,scores,accepted=cache
        r=scores.softmax(-1) if mode=='gaussian' else torch.nn.functional.one_hot(scores.argmax(-1),self.groups).float()
        r=r*accepted[:,:,None]
        self.count.add_(r.sum(0))
        self.total.add_(torch.einsum('bhk,bhd->hkd',r,z))
        self.square.add_(torch.einsum('bhk,bhd->hkd',r,z.square()))
        self.refresh()


class RoutedBDH(BDHWrapper):
    def __init__(self,spec,mode='sum',side=14):
        super().__init__(spec)
        self.mode=mode
        self.router=OnlineRouter(side*side,spec.n_head) if mode in ('centroid','gaussian') else None
        self.last_assignments=None
        self.pending=None

    def forward(self,idx,images=None):
        core=self.core
        B,T=idx.shape
        gate=None
        if self.router is not None:
            gate,self.last_assignments,self.pending=self.router.route(images,self.mode)
        x=core.ln(core.embed(idx).unsqueeze(1))
        def project(unit,x,weight):
            if gate is None: return unit(x,weight)
            a=unit.branch_activations(x)
            return unit.kwta((a*gate[:,:,None,:,None]).sum(-2))
        for _ in range(core.config.n_layer):
            q=project(self.neuron_unit_x,x,core.encoder)
            y=core.ln(core.attn(Q=q,K=q,V=x))
            v=project(self.neuron_unit_y,y,core.encoder_v)
            xy=core.drop(q*v)
            mlp=xy.transpose(1,2).reshape(B,1,T,-1)@core.decoder
            x=core.ln(x+core.ln(mlp))
        return x.view(B,T,-1)@core.lm_head

    def update_router(self):
        if self.router is not None:
            self.router.update(self.pending,self.mode)
            self.pending=None
