"""Functional branch-memory ablations and generic context-conditioned experts."""
import argparse,copy,json,statistics,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.deep_predictive_memory_pilot import Memory
from bdh_cl.scripts.shared_context_memory_pilot import Moments
from bdh_cl.scripts.usefulness_retention_synthetic import fit,audit,Window
from bdh_cl.scripts.electricity_memory_pilot import Forecast
from bdh_cl.scripts.electricity_usefulness_pilot import cohort,run
from bdh_cl.scripts.electricity_retention_pilot import audit as real_audit

MODES=('full','central_exact','common_cue','erase_hidden','shuffle_hidden','dense_gradient','uniform_output','moe_sparse','moe_dense','mlp','gru')

class DenseCredit(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,w):
        a=torch.tanh(q);ctx.save_for_backward(a);return a.gather(2,w[:,:,None]).squeeze(-1)
    @staticmethod
    def backward(ctx,grad):
        (a,)=ctx.saved_tensors;return grad[:,:,None]*(1-a.square())/4,None

class Mechanism(nn.Module):
    def __init__(self,mode,d=2,streams=16):
        super().__init__();self.mode=mode;self.d=d
        self.first=nn.Linear(d,32);self.second=nn.Linear(8,32)
        if mode=='central_exact':
            self.register_buffer('bank',torch.zeros(streams,128));self.register_buffer('seen',torch.zeros(32,dtype=torch.bool))
        elif mode=='common_cue':self.context=Moments(d,streams=streams);self.register_buffer('error',torch.ones(streams,32))
        else:self.context=Memory(32,streams);self.register_buffer('error',torch.ones(streams,32))
    @property
    def feedback(self):return self.mode!='uniform_output'
    def error_state(self):return self.bank[:,96:128] if self.mode=='central_exact' else self.error
    def reset(self):
        if self.mode=='central_exact':self.bank.zero_();self.seen.zero_();self.bank[:,96:].fill_(1)
        else:self.context.reset();self.error.fill_(1)
        self.recent=[]
    def input(self,x):return x.reshape(-1,8,2)[:,-1] if self.d==2 and x.shape[-1]==16 else x
    def first_scores(self):
        if self.mode=='central_exact':return self.bank[:,32:64]/(self.bank[:,64:96]+.01)
        if self.mode=='common_cue':return self.context.scores(self.first.weight,self.first.bias)
        score=self.context.cross/(self.context.energy+.01)
        if self.mode=='erase_hidden':return torch.zeros_like(score)
        if self.mode=='shuffle_hidden':return score.reshape(-1,8,4).roll(1,-1).flatten(1)
        return score
    def hidden(self,q,w):
        q=q.reshape(-1,8,4)
        return DenseCredit.apply(q,w) if self.mode=='dense_gradient' else torch.tanh(q.gather(2,w[:,:,None]).squeeze(-1))
    def forecast(self,x):
        x=self.input(x);q=self.first(x);w=self.first_scores().reshape(-1,8,4).argmax(-1);h=self.hidden(q,w);heads=self.second(h)
        weights=torch.full_like(heads,1/32) if self.mode=='uniform_output' else torch.softmax(-self.error_state()/.1,-1)
        pred=(heads*weights).sum(-1)
        with torch.no_grad():
            if self.mode=='central_exact':
                previous=self.bank[:,:32];cross=self.bank[:,32:64];energy=self.bank[:,64:96]
                cross.mul_(.95).add_(previous*q*.05*self.seen);energy.mul_(.95).add_(previous.square()*.05*self.seen);previous.copy_(q);self.seen.fill_(True)
            elif self.mode=='common_cue':self.context.write(x)
            else:self.context.write(q.detach())
        return pred,(x.detach().clone(),w.clone(),weights.detach().clone(),heads.detach().clone())
    @torch.no_grad()
    def observe(self,ticket,target):
        if self.feedback:self.error_state().mul_(.95).add_((ticket[3]-target[:,None]).square(),alpha=.05)
    def ticket_heads(self,ticket):return self.second(self.hidden(self.first(ticket[0]),ticket[1]))
    def ticket_prediction(self,ticket):return (self.ticket_heads(ticket)*ticket[2]).sum(-1)
    def training_loss(self,ticket,target):
        heads=self.ticket_heads(ticket);return F.mse_loss(heads,target[:,None].expand_as(heads))

class GenericExperts(nn.Module):
    def __init__(self,mode,d=2,streams=16):
        super().__init__();self.mode=mode;self.d=d;budget=32*(d+1)+288;self.n=budget//(3*(2*d+1)+4)
        self.first=nn.Linear(2*d,3*self.n);self.out=nn.Parameter(torch.empty(self.n,3));self.bias=nn.Parameter(torch.empty(self.n))
        nn.init.uniform_(self.out,-1/math.sqrt(3),1/math.sqrt(3));nn.init.uniform_(self.bias,-1/math.sqrt(3),1/math.sqrt(3))
        self.context=Memory(d,streams);self.register_buffer('error',torch.ones(streams,self.n))
    @property
    def feedback(self):return True
    def reset(self):self.context.reset();self.error.fill_(1);self.recent=[]
    def heads(self,z):return (torch.tanh(self.first(z).reshape(-1,self.n,3))*self.out).sum(-1)+self.bias
    def forecast(self,x):
        if self.d==2 and x.shape[-1]==16:x=x.reshape(-1,8,2)[:,-1]
        # Shared, causal input correlations condition every generic expert; no branch traces.
        cue=self.context.cross/(self.context.energy+.01);z=torch.cat((x,cue),1);heads=self.heads(z)
        weights=F.one_hot(self.error.argmin(-1),self.n).float() if self.mode in ('moe_sparse','moe_dense_hard') else torch.softmax(-self.error/.1,-1)
        with torch.no_grad():self.context.write(x)
        return (heads*weights).sum(-1),(z.detach().clone(),weights.detach().clone(),heads.detach().clone())
    @torch.no_grad()
    def observe(self,ticket,target):self.error.mul_(.95).add_((ticket[2]-target[:,None]).square(),alpha=.05)
    def ticket_prediction(self,ticket):return (self.heads(ticket[0])*ticket[1]).sum(-1)
    def training_loss(self,ticket,target):
        if self.mode=='moe_sparse':return F.mse_loss(self.ticket_prediction(ticket),target)
        q=self.heads(ticket[0]);return F.mse_loss(q,target[:,None].expand_as(q))

def make(mode,d):
    if mode in ('mlp','gru'):return Window(mode) if d==2 else Forecast(mode)
    return GenericExperts(mode,d) if mode.startswith('moe') else Mechanism(mode,d)

def toy(args,out):
    records=[]
    for seed in map(int,args.seeds.split(',')):
        for first in (0,1):
            for mode in args.modes.split(','):
                torch.manual_seed(seed);m=make(mode,2);m.reset();opt=torch.optim.Adam(m.parameters(),lr=.003)
                snap=fit(m,opt,seed+1000,(first,1-first),4096,capture=True)
                afterB=copy.deepcopy(m.state_dict());phases=audit(m,seed+5000,(first,1-first)*2)
                torch.save({'afterA':snap['model'],'afterB':afterB},out/f'{seed}_{first}_{mode}.pt')
                m.load_state_dict(snap['model']);before=audit(m,seed+5000,(first,1-first)*2)
                old=lambda rows:statistics.mean(p['all_mse'] for p in rows if p['regime']==first)
                new=statistics.mean(p['all_mse'] for p in phases if p['regime']!=first)
                r={'seed':seed,'first':first,'mode':mode,'old_before':old(before),'old_after':old(phases),'new_after':new,'forgetting':old(phases)-old(before),'afterA':before,'afterB':phases,
                   'parameters':sum(p.numel() for p in m.parameters()),'runtime_bytes':sum(b.numel()*b.element_size() for b in m.buffers())}
                records.append(r);(out/'results.json').write_text(json.dumps(records,indent=2));print('toy',seed,first,mode,round(r['old_after'],3),round(new,3),flush=True)
    return records

def real(args,out):
    A=cohort(0);B=cohort(16);assert not set(A['clients'])&set(B['clients'])
    stop=int(((A['year']==2012)&(A['month']==10)).nonzero()[0][0]);end=int((A['year']==2013).nonzero()[0][0]);records=[]
    (out/'cohorts.json').write_text(json.dumps({'A':A['clients'],'B':B['clients'],'train':'same2012prefix720:Oct1 forboth;heldoutOct-Dec2012;perclienttrain-prefixscalers'},indent=2))
    for seed in map(int,args.seeds.split(',')):
        for mode in args.modes.split(','):
            torch.manual_seed(seed);m=make(mode,25);m.reset();opt=torch.optim.Adam(m.parameters(),lr=.001)
            run(m,opt,A,720,stop,True,False);state=copy.deepcopy(m.state_dict());optstate=copy.deepcopy(opt.state_dict())
            before=real_audit(m,A,stop,end);newbefore=real_audit(m,B,stop,end)
            torch.save({'model':state,'optimizer':optstate},out/f'{seed}_{mode}_afterA.pt')
            run(m,opt,B,720,stop,True,False);after=real_audit(m,A,stop,end);newafter=real_audit(m,B,stop,end)
            torch.save(m.state_dict(),out/f'{seed}_{mode}_afterB.pt')
            r={'seed':seed,'mode':mode,'old_before':before['mse'],'old_after':after['mse'],'new_after':newafter['mse'],'new_before':newbefore['mse'],'forgetting':after['mse']-before['mse'],
               'afterA_old':before,'afterB_old':after,'afterB_new':newafter,'parameters':sum(p.numel() for p in m.parameters()),'runtime_bytes':sum(b.numel()*b.element_size() for b in m.buffers())}
            records.append(r);(out/'results.json').write_text(json.dumps(records,indent=2));print('real',seed,mode,'forget',round(r['forgetting'],4),'new',round(r['new_after'],4),flush=True)
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--kind',choices=['toy','real'],required=True);p.add_argument('--seeds',default='13,14,15');p.add_argument('--modes',default=','.join(MODES));args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'selection':'fixed ablations, no settings tuned on old/new heldout; newinitializations13-15',
      'full':'first-layer private response predictability + hard selection, all-outputtraining, soft outputerror memory(.95,.1)',
      'central_exact':'identical128float/clientstate packed centrally; preservesbranchidentity; exactfunctional control ofphysicalstorage location,not a testofremovingbranch-indexedinformation',
      'common_cue':'full shared input lag/covariance matrix read through learnedbranchweights; sameoutputs/errorrule;different stableinputcoordinates;toy46float+mass,real1357float+mass vsfull128',
      'erase_hidden':'no hidden context; firstbranchchosen; training remains same objective; reducedeffective capacity inherent inthis ablation',
      'shuffle_hidden':'cyclic shift memoryscores withinfour-branchgroup, breaking score-response correspondence without reducingstate orparameters',
      'dense_gradient':'same hard forwardselection, custombackward distributes hiddencredit across4branches; gradient sum scale matched but chain-gradient differs',
      'uniform_output':'same all-headgradienttraining, outputerror cue unused; equalweights outputs; isolatesquality readout from trainingcoverage',
      'genericexperts':'MLP experts conditioned oncurrentinput andshared diagonal EMAinputlag correlations; no branch-response traces. Winnergradient or all-expertgradient. Toy380params/20experts;real1099params/7experts; differentarchitecture but nearlyequalbudget, lowerfaststate.',
      'feedback':'causal targetarrival only, toy1step versusreal24h; slowweightsfrozeninaudits,fastcontext canlearn; no taskIDs/boundarymodelinputs; nooptimizer/fastreset acrossA/B',
      'real':'real measuredload only, no signal/label transformations. Sequential customer-domain shift Aoriginal16,Bdisjoint16, not fullychronological singlecustomerlife. Normalization/customerlanes known todata pipeline; no customerIDmodelfeature. Sameheldout periods before/afterB. No claimofgeneralization fromsame-dataset diagnosticalone.',
      'limits':'state/compute budgets reported not exactlymatched; central_exact has identical information byconstruction; dense_gradient changescreditassignment aswell asprotection; no unique-biologicalnecessityproof possiblefromthese comparisons'},indent=2))
    rows=toy(args,out) if args.kind=='toy' else real(args,out)
    lines=['# Branch mechanism ablations','','Positive forgetting means older heldout error increased after newer learning. Slowweights frozen in allprobes.','','| Model | Params | Old before | Old after | Forgetting | New after |','|---|---:|---:|---:|---:|---:|']
    for mode in args.modes.split(','):
        rs=[r for r in rows if r['mode']==mode];values=[statistics.mean(r[k] for r in rs) for k in ('old_before','old_after','forgetting','new_after')]
        lines.append(f'| {mode} | {rs[0]["parameters"]} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
