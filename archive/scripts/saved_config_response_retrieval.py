"""Input/response-only retrieval of five historical routes at frozen final weights."""
import json,time,statistics as st
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.split_mnist_usefulness import make
SOURCE=Path('results/split_mnist_usefulness_hard_20261011')
OUT=Path('results/saved_config_response_retrieval_20261014')
def save(p,x):p.write_text(json.dumps(x,indent=2))
@torch.no_grad()
def logits_all(model,x,routes):
 output=[]
 for bx in x.split(512):
  b=torch.tanh(model.first(bx)).reshape(-1,24,4)
  h=b[:,torch.arange(24)[None,:],routes]
  output.append(F.linear(h,model.head.weight,model.head.bias))
 return torch.cat(output)

def predict(logits,method,temp):
 prob=(logits/temp).softmax(-1);choice=None
 if method=='mean_probability':score=prob.mean(1)
 elif method=='mean_logit':score=logits.mean(1)
 else:
  top=prob.topk(2,-1).values
  if method=='max_probability':criterion=top[:,:,0]
  elif method=='probability_margin':criterion=top[:,:,0]-top[:,:,1]
  elif method=='low_entropy':criterion=(prob*prob.clamp_min(1e-12).log()).sum(-1)
  elif method=='max_logit':criterion=logits.max(-1).values
  else:raise ValueError(method)
  choice=criterion.argmax(1);score=prob[torch.arange(len(prob)),choice]
 return score.argmax(-1),choice

def metrics(pred,y,choice=None):
 pair=[float((pred[torch.isin(y,torch.tensor(p))]==y[torch.isin(y,torch.tensor(p))]).float().mean()) for p in SPLITS]
 result={'accuracy':float((pred==y).float().mean()),'pair_accuracy':pair}
 if choice is not None:
  correct_pair=y//2;result['oracle_pair_route_agreement']=float((choice==correct_pair).float().mean());result['selected_route_fraction']=torch.bincount(choice,minlength=5).float().div(len(y)).tolist()
 return result

class Selector(nn.Module):
 def __init__(self,fit_logits):
  super().__init__();z=fit_logits.flatten(1);self.register_buffer('center',z.mean(0));self.register_buffer('scale',z.std(0).clamp_min(.01));self.linear=nn.Linear(50,5);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
 def weights(self,logits):return self.linear((logits.flatten(1)-self.center)/self.scale).softmax(-1)
 def mixture(self,logits,temp):return (self.weights(logits)[:,:,None]*(logits/temp).softmax(-1)).sum(1)
 def loss(self,logits,y,temp):return -self.mixture(logits,temp)[torch.arange(len(y)),y].clamp_min(1e-12).log().mean()
 @torch.no_grad()
 def prediction(self,logits,temp,hard=False):
  w=self.weights(logits);choice=w.argmax(1)
  scores=(logits/temp).softmax(-1)[torch.arange(len(logits)),choice] if hard else self.mixture(logits,temp)
  return scores.argmax(-1),choice

def train_selector(logits,y,vl,vy,lr,temp,seed,epochs=20):
 m=Selector(logits);opt=torch.optim.Adam(m.parameters(),lr=lr);history=[]
 for epoch in range(epochs):
  rng=torch.Generator().manual_seed(seed*1000+epoch)
  for ix in torch.randperm(len(y),generator=rng).split(512):
   opt.zero_grad(set_to_none=True);m.loss(logits[ix],y[ix],temp).backward();opt.step()
  if epoch+1 in [5,10,20]:
   pred,_=m.prediction(vl,temp);history.append({'epoch':epoch+1,'validation_accuracy':float((pred==vy).float().mean())})
 return m,history

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False);(OUT/'selectors').mkdir();rows=[];source=json.loads((SOURCE/'results.json').read_text())
 save(OUT/'protocol.json',{'question':'Can five saved branch configurations recover accuracy through input/response-only selection without oracle pair information?',
 'source':str(SOURCE),'models':'Originalhardlocal24x4network,finalweights/headfrozen,seeds34/35/36. NaturalforwardSplitMNISTorder,twoAdam.001passes/pair. Five24-indexroutesrecordedafterstages. No retrainingoforiginalnetwork.',
 'inference':'Computeallfive routes on eachinput;eachrouteemitsalltenlogits. No pair/classmask, testlabel or oracleIDusedbycandidategeneration/selection. Sharedfirstlayerresponsescomputedonce. Existingfinal route ordinarybaseline; oraclepair selection ONLYdiagnostic afterscoresmade.',
 'calibration':'60000originaltrainingimages:fixedstratified1000/class validation for baseline/selector,remaining50000for fittingselector. These imageswerepreviouslyusedbytheoriginalencoder: retrospective training-image calibration,notstrictindependentencoderholdout. Official10000testlabels scoreonlyafter settingschosen. No testtuning.',
 'fixed_rules':'maxprobability, probabilitymargin, lowentropy, meanprobability overtemperatures.25/.5/1/2/4;maxrawlogitandmeanrawlogit. Allcandidatesarchived;onebestvalidationrule chosenbeforeofficialtest. Tiesfirstcandidate. Temperature1fixedbaselinesalso reported.',
 'learned_selector':'50networklogits standardizedusingfit-onlycenter/scale;linear50->5softmax weights;mixtureofcandidateallten-classprobabilities. Trainedbylabelnegative-loglikelihood,not task/route IDs. Finalnetworkandroutesunchanged. lr.003/.03,T.5/1/2,20epochs;choose validationmixtureaccuracy. Hardrouteprediction also reported atsamechosenmodel withoutretuning. Thisisadditionalcentralrouterwithparameters,notbranch-localmemory.',
 'memory':'Original75610FP32parametersplus96FP32error=302824bytes. Five24-indexint64routes960bytes. Rule retrievalbase303784bytes. Selector255FP32params1020bytesplus50center/50scale400bytes;total305204bytesexcludingscalarmetadata. Adamstateallocatedduringroutertraining,countedseparately;noneatdeployment. Computeusesallroutes,nostoredexamplesatdeployment.',
 'limits':'Originalthreeseedsandoneclassorder,posthocexternallysavedroutes. Successfulretrievaldoesnotdemonstrateautonomousconfigstorageorlocalbiologicalmechanism. Selectorhaslabeledtrainingfeedbackbutnotinferencefeedback. Groundtruthpaironlyoracleanalyses.'})
 (OUT/'source.py').write_text(Path(__file__).read_text());raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long);tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
 rng=torch.Generator().manual_seed(481);val=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)]);fit=torch.ones(len(y),dtype=torch.bool);fit[val]=False
 for seed in [34,35,36]:
  oldrow=next(r for r in source if r['seed']==seed and r['arm']=='local' and r['mode']=='sequential');m=make('local',gate='hard');m.load_state_dict(torch.load(SOURCE/f'{seed}_local_sequential.pt',weights_only=True));before={k:v.clone() for k,v in m.state_dict().items()};routes=torch.tensor([s['preferred_branches'] for s in oldrow['stages']]);t=time.perf_counter();train_logits=logits_all(m,x,routes);test_logits=logits_all(m,tx,routes);candidate_seconds=time.perf_counter()-t
  assert test_logits.shape==(10000,5,10)
  # Independent originalmodelforward with restoredroute must agree.
  error=m.error.clone()
  for k in range(5):
   m.error.fill_(1);m.error.scatter_(1,routes[k,:,None],0);expected=torch.cat([m(b) for b in tx.split(512)])
   assert torch.allclose(expected,test_logits[:,k],atol=2e-5,rtol=1e-5)
  m.error.copy_(error)
  current=m.gates().argmax(-1);same=(routes==current[None]).all(-1).nonzero().flatten();assert len(same)>0;ordinary=metrics(test_logits[:,same[-1]].argmax(-1),ty);assert abs(ordinary['accuracy']-oldrow['final_accuracy'])<1e-6
  oracle_pred=test_logits[torch.arange(len(ty)),ty//2].argmax(-1);oracle=metrics(oracle_pred,ty);any_correct=(test_logits.argmax(-1)==ty[:,None]).any(-1).float().mean().item()
  candidates=[]
  for method in ['max_probability','probability_margin','low_entropy','mean_probability','max_logit','mean_logit']:
   for temp in ([1.] if method in ['max_logit','mean_logit'] else [.25,.5,1.,2.,4.]):
    pred,_=predict(train_logits[val],method,temp);candidates.append({'method':method,'temperature':temp,'validation_accuracy':float((pred==y[val]).float().mean())})
  chosen=max(candidates,key=lambda r:r['validation_accuracy']);rule_results=[]
  for method in ['max_probability','probability_margin','low_entropy','mean_probability','max_logit','mean_logit']:
   pred,choice=predict(test_logits,method,1.);rule_results.append({'method':method,'temperature':1.,**metrics(pred,ty,choice)})
  pred,choice=predict(test_logits,chosen['method'],chosen['temperature']);selected_rule={**chosen,**metrics(pred,ty,choice)}
  selectors=[];models=[];t=time.perf_counter()
  for lr in [.003,.03]:
   for temp in [.5,1.,2.]:
    router,h=train_selector(train_logits[fit],y[fit],train_logits[val],y[val],lr,temp,seed);selectors.append({'lr':lr,'temperature':temp,'validation_accuracy':h[-1]['validation_accuracy'],'history':h});models.append(router)
  ix=max(range(len(selectors)),key=lambda i:selectors[i]['validation_accuracy']);router=models[ix];setting=selectors[ix];soft,weights_choice=router.prediction(test_logits,setting['temperature']);hard,hard_choice=router.prediction(test_logits,setting['temperature'],True)
  soft_metrics=metrics(soft,ty);soft_metrics['max_weight_route_agreement_analysis_only']=float((weights_choice==ty//2).float().mean());hard_metrics=metrics(hard,ty,hard_choice)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
  selector_params=sum(p.numel()*p.element_size() for p in router.parameters());selector_buffers=sum(p.numel()*p.element_size() for p in router.buffers());basebytes=sum(v.numel()*v.element_size() for v in m.state_dict().values())+routes.numel()*routes.element_size();assert basebytes==303784 and selector_params+selector_buffers==1420
  torch.save({'selector':router.state_dict(),'routes':routes,'temperature':setting['temperature']},OUT/'selectors'/f'{seed}.pt')
  row={'seed':seed,'ordinary':ordinary,'oracle_pair_route':oracle,'any_candidate_correct_label_diagnostic':any_correct,'fixed_rules':rule_results,'rule_validation_candidates':candidates,'validation_selected_rule':selected_rule,'selector_validation_candidates':selectors,'selector_setting':setting,'selector_soft':soft_metrics,'selector_hard':hard_metrics,'route_bank_bytes':basebytes,'deployed_selector_bytes':basebytes+selector_params+selector_buffers,'candidate_prediction_seconds_train_and_test':candidate_seconds,'selector_training_seconds':time.perf_counter()-t,'routes':routes.tolist()};rows.append(row);save(OUT/'results.json',rows)
  print(seed,'ordinary',round(ordinary['accuracy']*100,2),'oracle',round(oracle['accuracy']*100,2),'rule',chosen['method'],round(selected_rule['accuracy']*100,2),'selector_soft',round(soft_metrics['accuracy']*100,2),'hard',round(hard_metrics['accuracy']*100,2),flush=True)
 save(OUT/'verification.json',{'complete_3_seeds':len(rows)==3,'original_forward_matches_all_5_candidate_logits':True,'all_ten_logits_no_pair_mask':True,'original_weights_head_error_unchanged':True,'ordinary_prediction_reproduces_checkpoint_result':True,'numeric_state_matches_protocol':True})
if __name__=='__main__':main()
