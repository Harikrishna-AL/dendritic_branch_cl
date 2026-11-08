"""Isolate grouped training competition, protection and compatibility decoding.

No claim that naming grouped hidden units 'dendrites' distinguishes an equivalent
ordinary implementation. Global-training arm keeps structured inference to
isolate training competition; unstructured readout is reported separately.
"""
import argparse,json,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_plasticity import LocalVariant
from bdh_cl.scripts.local_graph_resource_benchmark import permutations,dump
from bdh_cl.scripts.local_graph_domains import branch_scores

class Mechanism(LocalVariant):
 def __init__(self,seed,s):
  super().__init__(seed,s);self.competition=s['competition']
  if self.competition=='dense':self.links=torch.empty(0)
 def select(self):
  if self.competition=='grouped':return super().select()
  if self.competition=='dense':return torch.arange(256)
  error=self.error.flatten();chosen=error.topk(64,largest=False).indices
  if float(error[chosen].max())>self.threshold:chosen=self.usage.topk(64,largest=False).indices
  return chosen
 @torch.no_grad()
 def learn(self,x,y):
  z=self.data_features(x);b=torch.tanh(z@self.weight);target=self.codes[y].repeat_interleave(4,dim=1);active=self.select();residual=target[:,active]-b[:,active];reliable=(residual.abs()<.2).float()
  if self.competition!='dense':
   signals=torch.zeros(len(x),256).scatter_(1,active[None].expand(len(x),-1),reliable).reshape(-1,64,4)
   self.links+=torch.einsum('bnk,bnl->nkl',signals[:,:-1],signals[:,1:])
  rate=torch.full_like(self.usage[active],self.rate) if self.protection=='off' else self.rate/(1+self.usage[active]/self.tau)
  self.weight[:,active]+=(z.T@(residual*(1-b[:,active].square())))/len(x)*rate[None]
  self.usage[active]+=reliable.sum(0);self.error.copy_((b-target).square().mean(0).reshape(64,4));self.presentations+=len(x)
 @torch.no_grad()
 def score(self,b,readout):
  if readout=='communicating':return branch_scores(self,b,.1)
  value=-(b[:,None]-self.codes.repeat_interleave(4,dim=1)[None]).square()
  if readout=='grouped_no_communication':return value.reshape(-1,10,64,4).max(-1).values.sum(-1)
  if readout=='unstructured_top64':return value.topk(64,dim=-1).values.sum(-1)
  # Equivalent to a conventional one-hidden-layer coded MLP with fixed linear output weights.
  return value.sum(-1)

@torch.no_grad()
def evaluate_all(model,x,y,ps,readouts):
 result={mode:[] for mode in readouts}
 for p in ps:
  predictions={mode:[] for mode in readouts}
  for bx in x.split(256):
   b=torch.tanh(model.data_features(bx[:,p])@model.weight)
   for mode in readouts:predictions[mode].append(model.score(b,mode).argmax(-1))
  for mode in readouts:result[mode].append(float((torch.cat(predictions[mode])==y).float().mean()))
 return result

def verify():
 torch.manual_seed(820);x=torch.rand(17,784);y=torch.arange(17)%10;s={'rate':1.,'tau':10000.,'gamma':.1,'gated':True,'loss':'mse','protection':'absolute','allocation':'novelty','threshold':.4,'competition':'grouped'}
 a=LocalVariant(3,s);b=Mechanism(3,s)
 for _ in range(4):a.learn(x,y);b.learn(x,y)
 assert all(torch.equal(getattr(a,k),getattr(b,k)) for k in ['weight','codes','error','usage','links'])
 response=torch.tanh(b.data_features(x)@b.weight)
 assert torch.allclose(b.score(response,'grouped_no_communication'),branch_scores(b,response,0.),atol=2e-5,rtol=1e-5)
 # Same branches are an ordinary256-unit hidden layer plus a sparse selection matrix.
 chosen=b.select();routing=torch.zeros(256,64);routing[chosen,torch.arange(64)]=1.
 assert torch.equal(response[:,chosen],response@routing)
 linear=response@b.codes.repeat_interleave(4,dim=1).T
 assert torch.equal(linear.argmax(-1),b.score(response,'dense_fixed_readout').argmax(-1))
 return {'grouped_training_matches_prior_novelty_rule_exactly':True,'zero_communication_fast_decoder_matches_chain_energy':True,'ordinary_hidden_layer_with_same_routing_is_equivalent':True,'dense_fixed_readout_equals_linear_coded_output':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);dump(out/'verification.json',verify());rows=[]
 protocol={'question':'Do grouped training competition and learned local compatibility help beyond protection, with identical inputs/teaching/weights?','datasets':['mnist','fashion_mnist'],'seeds':{'mnist':[391,392,393],'fashion_mnist':[401,402,403]},'domains':16,'training':'Two passes/currentdomain,60kreal images; commonpermutationprefix andshuffle acrossarms; no replay, taskIDs, savedconfigs or newdata per arm. Settings locked before scoring:rate1,tau10000,threshold.4,gamma.1. No hyperparameter selection using these tests.', 'factorial':'Groupedor global64-unit trainingcompetition crossed with protectionabsolute/off; communicating andno-communication readouts scored on SAMEfinalweights. Same256inputresponse units,10x64codes expanded perunit,256error/usage state,63x4x4paircounts. Grouped/global train64units perbatch. Both retain identical structureddecoder for primarycomparison; this isolates trainingcompetition, NOT wholesale deletion of branch organization.', 'sparsegraph':'Global activeunits may be unevenly distributed across4-unit neighborhoods. Identical spatialpairedges record only reliablejointactivity. Target codes identical to groupedmodel. Uneven code coverage/redundancy is a possible explanation for groupbenefits, explicitly notproof of uniquelybiological dendrites.', 'additionalcontrols':'Global independent top64input-conditioned coded response readout, no graph or one-per-group constraint; dense256-unit coded MLP updates/readsallunits, same localfeedback andfixedoutput. Dense usesrate.25,tau40000 to compensate4xper-unit trainingexposure andnominalupdate count, notindependently optimized; largercompute acknowledged. Dense has808448bytes versus812480grouped/global, no storagepadding. These are restrictedlocalcoded controls; prior tunedordinaryMLP remains stronger broadcontrol.', 'metrics':'Full10ktest per domain final16; all-domain, old15, first4, recent4,newest. Inference labels onlyfor scoring. No statisticalsignificance claim from3pairedseeds.', 'interpretation':'A functional component advantage can be attributed to groupedcompetition/coactivity within this model. Equivalentordinaryunits plus identicalrouting compute samefunction, so this cannotprove anatomicaldendrites uniquelyretainmemory. No broadclaim of outperforming allordinaryMLPs.'}
 dump(out/'protocol.json',protocol);(out/'source.py').write_text(Path(__file__).read_text())
 for ds in ['mnist','fashion_mnist']:
  raw=load_raw('data_cache/'+ds);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
  for seed in protocol['seeds'][ds]:
   ps=permutations(seed,16)
   for competition in ['grouped','global','dense']:
    for protection in ['absolute','off']:
     s={'rate':.25 if competition=='dense' else 1.,'tau':40000. if competition=='dense' else 10000.,'gamma':.1,'gated':True,'loss':'mse','protection':protection,'allocation':'novelty','threshold':.4,'competition':competition};m=Mechanism(seed,s)
     for domain,p in enumerate(ps):
      for epoch in range(2):
       rng=torch.Generator().manual_seed(seed*100000+domain*100+epoch)
       for ix in torch.randperm(len(y),generator=rng).split(128):m.learn(x[ix][:,p],y[ix])
     readouts=['dense_fixed_readout'] if competition=='dense' else ['communicating','grouped_no_communication','unstructured_top64'];results=evaluate_all(m,tx,ty,ps,readouts)
     for mode,a in results.items():
      row={'dataset':ds,'seed':seed,'competition':competition,'protection':protection,'readout':mode,'accuracy':st.mean(a),'old_accuracy':st.mean(a[:-1]),'first_four':st.mean(a[:4]),'recent_four':st.mean(a[-4:]),'newest':a[-1],'domain_accuracy':a,'numeric_bytes':m.bytes(),'setting':s};rows.append(row);dump(out/'results.json',rows);print(ds,seed,competition,protection,mode,'all',round(row['accuracy']*100,2),'first4',round(row['first_four']*100,2),'new',round(row['newest']*100,2),flush=True)
 report(out,rows)

def report(out,rows):
 lines=['# Grouping, protection and local communication: controlled mechanism tests','','Three fresh paired seeds per dataset;16domains, fixed localfeedback andinputweights. Grouped/global arms differ in TRAININGcompetition, with identicalstructuredinference in primarycomparisons. Densecoded controls differ in readout andactivecount; they do not substitute for independentlytuned ordinaryMLP benchmarks.','', '| Dataset | Training competition | Protection | Readout | Allaccuracy | First4 | Newest |','|---|---|---|---|---:|---:|---:|']
 for ds in ['mnist','fashion_mnist']:
  for comp in ['grouped','global','dense']:
   for protection in ['absolute','off']:
    for mode in (['dense_fixed_readout'] if comp=='dense' else ['communicating','grouped_no_communication','unstructured_top64']):
     rs=[r for r in rows if (r['dataset'],r['competition'],r['protection'],r['readout'])==(ds,comp,protection,mode)];f=lambda k:100*st.mean(r[k] for r in rs)
     lines.append(f'| {ds} | {comp} | {protection} | {mode} | {f("accuracy"):.2f}% | {f("first_four"):.2f}% | {f("newest"):.2f}% |')
 lines+=['','Ordinary256-unit responses with identicalgating/routing are numerically equivalent to this branch model. Results support atmost computationalgrouping/competition/communication benefits, not biological uniqueness. Codecoverage, decoderconstraints andlocalteaching are retainedlimits.'];(out/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
