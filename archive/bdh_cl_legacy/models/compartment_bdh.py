"""Dendritic BDH with a bounded, position-free branch-indexed slow state.

All inputs are image prompts only. Shared and uniform compartment memory are
algebraically identical controls; selective read/write access is the treatment.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from .bdh_wrapper import BDHWrapper


class CompartmentBDH(BDHWrapper):
    def __init__(self, spec, mode='none', decay=.999, eta=.01, read_scale=.1):
        super().__init__(spec)
        if mode not in ('none','shared','uniform','selective'):
            raise ValueError(mode)
        self.mode, self.decay, self.eta, self.read_scale = mode, decay, eta, read_scale
        self.k = self.neuron_unit_x.k
        self.m = spec.N // self.k
        gen = torch.Generator().manual_seed(1234)
        # Same fixed compression in every arm, retaining branch identity.
        projection = torch.randn(spec.N, self.m, generator=gen) / math.sqrt(spec.N)
        self.register_buffer('compression', projection)
        self.router = nn.Parameter(torch.randn(spec.n_head,self.k,self.m,generator=gen)*.1)
        if mode != 'none':
            self.register_buffer('slow_state', torch.zeros(spec.n_head,self.k,self.m,spec.n_embd))
        else:
            self.slow_state = None
        self.last_routing = None
        self.last_head_routing = None
        self.last_write_choices = None
        self.last_read_ratio = None

    def memory_bytes(self):
        return 0 if self.slow_state is None else self.slow_state.numel()*self.slow_state.element_size()

    def forward(self, idx, targets=None, *, update_memory=False, return_activations=False,
                return_hidden=False, loss_mask=None):
        core = self.core
        B,T = idx.shape
        D,nh,N = self.spec.n_embd,self.spec.n_head,self.spec.N
        x = core.ln(core.embed(idx).unsqueeze(1))
        snap = self.slow_state.detach().clone() if self.slow_state is not None else None
        staged = []
        routes, read_ratios, choices = [], [], []
        for _ in range(self.spec.n_layer):
            branch = self.neuron_unit_x.branch_activations(x)
            q = self.neuron_unit_x.pool(branch)
            q = self.neuron_unit_x.kwta(q)
            fast = core.attn(Q=q,K=q,V=x)
            if snap is not None:
                # Compress each branch independently; preserve branch index.
                code = torch.einsum('bhtkn,nm->bhtkm', branch, self.compression)
                # Sensory routing only; no labels, task IDs, or recent-task signal.
                summary = F.normalize(code.mean(dim=2),dim=-1)
                route = (5.*torch.einsum('bhkm,hkm->bhk',summary,self.router)).softmax(-1)
                routes.append(route.detach().mean(dim=0))
                choices.append(F.one_hot(route.detach().argmax(-1),self.k).float().mean(dim=0))
                if self.mode == 'selective':
                    read_gate = self.k * route
                    write_gate = self.k * F.one_hot(route.argmax(-1),self.k).float()
                else:
                    read_gate = write_gate = torch.ones_like(route)
                gated_code = code * read_gate[:,:,None,:,None]
                if self.mode == 'shared':
                    recalled = torch.einsum('bhtf,hfd->bhtd',gated_code.flatten(-2),snap.flatten(1,2))
                else:
                    recalled = torch.einsum('bhtkm,hkmd->bhtd',gated_code,snap)
                # Bound recalled contribution relative to the within-image path.
                # Raw slow reads can otherwise be negligible beside unnormalized attention.
                gain = fast.detach().norm(dim=-1,keepdim=True) / recalled.detach().norm(dim=-1,keepdim=True).clamp_min(1e-6)
                recalled = self.read_scale * gain * recalled
                read_ratios.append(float(recalled.detach().norm()/fast.detach().norm().clamp_min(1e-8)))
                if update_memory:
                    v=x.expand(-1,nh,-1,-1)
                    writecode=code.detach()*write_gate.detach()[:,:,None,:,None]
                    staged.append(torch.einsum('bhtkm,bhtd->hkmd',writecode,v.detach())/(B*T))
                fast=fast+recalled
            y=core.ln(fast)
            ys=self.neuron_unit_y(y,core.encoder_v)
            xy=core.drop(q*ys)
            mlp=xy.transpose(1,2).reshape(B,1,T,N*nh)@core.decoder
            x=core.ln(x+core.ln(mlp))
        if staged:
            with torch.no_grad():
                self.slow_state.mul_(self.decay).add_(torch.stack(staged).mean(0),alpha=self.eta)
        self.last_routing=torch.stack(routes).mean(dim=(0,1)).cpu().tolist() if routes else None
        self.last_head_routing=torch.stack(routes).mean(0).cpu().tolist() if routes else None
        self.last_write_choices=torch.stack(choices).mean(0).cpu().tolist() if choices else None
        self.last_read_ratio=sum(read_ratios)/len(read_ratios) if read_ratios else 0.
        hidden=x.view(B,T,D)
        logits=hidden@core.lm_head
        if return_hidden:
            return logits,None,hidden
        return logits,None
