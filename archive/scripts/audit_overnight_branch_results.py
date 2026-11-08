"""Completeness, memory, source hashes and independent saved-state prediction audit."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts import online_proxy_branch_retrieval as e
from bdh_cl.data.mnist import load_raw
ROOT=Path('results/overnight_branch_research_20261024')
def main():
 torch.set_num_threads(2);audit={'groups':{},'independent_predictions':[],'file_sha256':{}};total=0
 for name,start,expected in [('multicue_confirmation',562,120),('pmi_confirmation',602,90),('pmi_fashion_transfer',610,90)]:
  p=ROOT/name;done=json.loads((p/'completion.json').read_text());runs=json.loads((p/'results.json').read_text());assert done['complete'];assert {(r['seed'],r['order']) for r in runs}=={(seed,order) for seed in range(start,start+5) for order in ['forward','reverse']};n=sum(len(r['methods']) for r in runs);assert n==expected;total+=n
  for r in runs:
   assert r['diagnostics']['evaluation_state_unchanged'];assert r['diagnostics']['model_bytes']==315440;assert r['write_audit'][-1]['accepted']['1']==120000
   ck=torch.load(p/'models'/f"{r['seed']}_{r['order']}.pt",weights_only=True);assert sum(v.numel()*v.element_size() for v in ck['state_dict'].values())==315440
   with np.load(p/'memories'/f"{r['seed']}_{r['order']}.npz") as a:
    if name!='multicue_confirmation':assert a['stable1_cooc'].nbytes==2560 and np.all(a['stable1_cooc'].sum((1,2))==120000)
  audit['groups'][name]={'backbones':len(runs),'readouts':n,'all_seed_order_pairs_present':True,'evaluation_state_unchanged':True,'memory_and_120000_presentations_checked':True}
  for filename in ['protocol.json','source.py','results.json','completion.json']:
   path=p/filename;audit['file_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 for name,dataset,seed in [('pmi_confirmation','mnist',602),('pmi_fashion_transfer','fashion_mnist',610)]:
  p=ROOT/name;ck=torch.load(p/'models'/f'{seed}_forward.pt',weights_only=True)['state_dict'];raw=load_raw('data_cache/'+dataset);x=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;y=raw['test_labels'].copy();q=np.concatenate([e.keys(b) for b in x.split(512)])
  with np.load(p/'memories'/f'{seed}_forward.npz') as a:
   sums=a['stable1_d64_sums'];counts=a['stable1_d64_count'];score=np.einsum('bd,nksd->bnks',q,sums/np.maximum(np.linalg.norm(sums,axis=-1,keepdims=True),1e-8));score=np.where(counts[None]>0,score,-1e3).max(-1);route=score.argmax(-1);assert sums.nbytes+counts.nbytes==104000
  # Independent NumPy implementation of the factual branched forward and classifier.
  response=np.tanh(x.numpy()@ck['layers.0.weight'].numpy().T+ck['layers.0.bias'].numpy()).reshape(-1,25,4);h=np.take_along_axis(response,route[:,:,None],-1).squeeze(-1);logits=h@ck['head.weight'].numpy().T+ck['head.bias'].numpy();pred=logits.argmax(-1);accuracy=float((pred==y).mean());r=next(r for r in json.loads((p/'results.json').read_text()) if r['seed']==seed and r['order']=='forward');expected=next(z['accuracy'] for z in r['methods'] if z['method']=='no_message' and z['dimension']==64);assert abs(accuracy-expected)<=.0002
  m=e.backbone.make(seed,'branch1_proxy');m.load_state_dict(ck)
  with torch.no_grad():
   tb=torch.cat([torch.tanh(m.layers[0](bx)).reshape(-1,25,4) for bx in x.split(512)]);th=tb.gather(-1,torch.tensor(route)[:,:,None]).squeeze(-1);torchpred=m.head(th).argmax(-1).numpy()
  agreement=float((pred==torchpred).mean());assert agreement>=.9999;audit['independent_predictions'].append({'dataset':dataset,'seed':seed,'order':'forward','condition':'D64S4_no_message','numpy_accuracy':accuracy,'recorded_accuracy':expected,'numpy_torch_prediction_agreement':agreement,'prototype_numeric_bytes':104000})
  for filename in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']:
   path=Path('data_cache')/dataset/filename
   if path.exists():audit['file_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 audit.update(complete=True,confirmation_backbones=30,automatic_readouts=total,scope='Verification only. No training, setting changes, extra seeds or test-based selection. NumPy forward reimplements saved-state neural predictions; image-key extraction is shared with original implementation.');e.save(ROOT/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='file_sha256'},indent=2))
if __name__=='__main__':main()
