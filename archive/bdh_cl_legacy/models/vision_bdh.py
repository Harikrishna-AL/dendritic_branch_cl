"""Patch-input BDH with optional grouped value-side dendritic residuals."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from bdh_cl.models.bdh_wrapper import upstream_bdh


class BranchDistributions(nn.Module):
    def __init__(self, depth, heads, groups, branches, width, dim=16):
        super().__init__()
        self.groups,self.branches=groups,branches
        gen=torch.Generator().manual_seed(1234)
        self.register_buffer('projection',torch.randn(heads,width,dim,generator=gen)/math.sqrt(width))
        shape=(depth,heads,groups,branches,dim)
        for name in ('slow_mean','fast_mean'):
            self.register_buffer(name,torch.zeros(shape))
        for name in ('slow_second','fast_second'):
            self.register_buffer(name,torch.ones(shape))
        self.register_buffer('threshold',torch.zeros(depth,heads,groups))
        self.register_buffer('ready',torch.zeros(depth,dtype=torch.bool))
        self.pending=[]
        self.observations=[]

    def project(self,c):
        return torch.einsum('bhtd,hdf->bhtf',c.detach(),self.projection)

    def score(self,z,level):
        var=(self.slow_second[level]-self.slow_mean[level].square()).clamp_min(.02)
        delta=z[:,:,:,None,None,:]-self.slow_mean[level][None,:,None]
        return -.5*(delta.square()/var[None,:,None]+var[None,:,None].log()).sum(-1)

    @torch.no_grad()
    def initialize(self,c,level):
        # c: B,H,T,D. Bound fitting work; never use validation/test activations.
        z=self.project(c).permute(0,2,1,3).reshape(-1,c.shape[1],16)
        z=z[::max(1,len(z)//2048)][:2048]
        H=z.shape[1];K=self.branches
        for h in range(H):
            for g in range(self.groups):
                centers=[z[(g*37)%len(z),h]]
                for _ in range(1,K):
                    distances=(z[:,h,None]-torch.stack(centers)[None]).square().sum(-1).amin(-1)
                    centers.append(z[distances.argmax(),h])
                means=torch.stack(centers)
                for _ in range(8):
                    assignments=(z[:,h,None]-means[None]).square().sum(-1).argmin(-1)
                    for k in range(K):
                        part=z[assignments==k,h]
                        if len(part):means[k]=part.mean(0)
                for k in range(K):
                    part=z[assignments==k,h]
                    var=part.var(0,unbiased=False).clamp_min(.02) if len(part)>1 else torch.ones_like(means[k])
                    for speed in ('slow','fast'):
                        getattr(self,speed+'_mean')[level,h,g,k].copy_(means[k])
                        getattr(self,speed+'_second')[level,h,g,k].copy_(var+means[k].square())
        scores=self.score(z[:, :,None,:],level).squeeze(2).amax(-1)
        self.threshold[level].copy_(torch.quantile(scores,.01,dim=0))
        self.ready[level]=True

    @torch.no_grad()
    def forward(self,c,level):
        if not bool(self.ready[level]):raise RuntimeError('Uninitialized branch distributions')
        z=self.project(c);scores=self.score(z,level)
        best,choice=scores.max(-1)
        accepted=best>=self.threshold[level][None,:,None]
        gates=F.one_hot(choice,self.branches).float()*accepted[...,None]
        if self.training:
            # Save only sufficient statistics, not full activations, across depth.
            n=gates.sum((0,2))
            sums=torch.einsum('bhtgk,bhtd->hgkd',gates,z)
            seconds=torch.einsum('bhtgk,bhtd->hgkd',gates,z.square())
            self.pending.append((level,n,sums,seconds))
        self.observations.append(gates.mean((0,2)).detach())
        return gates

    @torch.no_grad()
    def commit(self):
        for level,n,sums,seconds in self.pending:
            mean=sums/n.clamp_min(1)[...,None]
            second=seconds/n.clamp_min(1)[...,None]
            valid=(n>0)[...,None]
            for speed,alpha in [('slow',.002),('fast',.05)]:
                m=getattr(self,speed+'_mean')[level];q=getattr(self,speed+'_second')[level]
                m.copy_(torch.where(valid,(1-alpha)*m+alpha*mean,m))
                q.copy_(torch.where(valid,(1-alpha)*q+alpha*second,q))
        self.pending.clear()

    @torch.no_grad()
    def metrics(self):
        if not self.observations:return {}
        usage=torch.stack(self.observations).mean(0)
        var=(self.slow_second-self.slow_mean.square()).clamp_min(.02)
        drift=((self.fast_mean-self.slow_mean).square()/var).mean().sqrt()
        result={'gate/rejection':float(1-usage.sum(-1).mean()),'gate/distribution_drift':float(drift)}
        for k in range(self.branches):result[f'gate/branch_{k}_usage']=float(usage[...,k].mean())
        return result


class VisionBDH(nn.Module):
    def __init__(self,width=128,neurons=512,heads=4,depth=4,classes=100,dropout=.1,branches=0,groups=4,mode='open'):
        super().__init__()
        assert neurons%groups==0
        self.width,self.neurons,self.heads,self.depth=width,neurons,heads,depth
        self.branches,self.groups,self.mode=branches,groups,mode
        self.patch=nn.Linear(48,width)
        self.cls=nn.Parameter(torch.randn(1,1,width)*.02)
        self.position=nn.Parameter(torch.randn(1,65,width)*.02)
        self.encoder=nn.Parameter(torch.randn(heads,width,neurons)*.02)
        self.encoder_v=nn.Parameter(torch.randn(heads,width,neurons)*.02)
        self.decoder=nn.Parameter(torch.randn(heads*neurons,width)*.02)
        self.ln=nn.LayerNorm(width,elementwise_affine=False)
        self.drop=nn.Dropout(dropout)
        cfg=upstream_bdh.BDHConfig(n_layer=depth,n_embd=width,n_head=heads,mlp_internal_dim_multiplier=neurons*heads//width)
        self.attn=upstream_bdh.Attention(cfg)
        self.classifier=nn.Linear(width,classes)
        if branches:
            self.branch_weight=nn.Parameter(torch.randn(heads,width,branches*neurons)*.02)
            # Zero output gains preserve the pretrained function; gains get gradients immediately.
            self.branch_gain=nn.Parameter(torch.zeros(heads,branches,neurons))
            self.router=BranchDistributions(depth,heads,groups,branches,width)
        else:self.router=None

    def forward(self,images,calibrate=False):
        B=len(images)
        patches=F.unfold(images,kernel_size=4,stride=4).transpose(1,2)
        x=torch.cat((self.patch(patches),self.cls.expand(B,-1,-1)),1)+self.position
        x=self.ln(x).unsqueeze(1)
        if self.router is not None:
            self.router.observations.clear()
        for level in range(self.depth):
            q=F.relu(x@self.encoder)
            c=self.ln(self.attn(Q=q,K=q,V=x))
            pre=c@self.encoder_v
            if self.branches:
                if calibrate:self.router.initialize(c,level)
                b=F.relu(c@self.branch_weight).view(B,self.heads,65,self.branches,self.neurons)
                if self.mode=='gaussian':
                    gates=self.router(c,level).permute(0,1,2,4,3).repeat_interleave(self.neurons//self.groups,dim=-1)
                    b=b*gates
                pre=pre+(b*self.branch_gain[None,:,None]).sum(-2)
            v=F.relu(pre)
            a=self.drop(q*v)
            y=a.transpose(1,2).reshape(B,1,65,-1)@self.decoder
            x=self.ln(x+self.ln(y))
        return self.classifier(x[:,0,-1])

    def commit(self):
        if self.router:self.router.commit()

    def set_backbone_trainable(self,trainable):
        for name,p in self.named_parameters():
            p.requires_grad_(trainable or name.startswith(('branch_','classifier.')))
