"""CPU inference latency diagnostic with shared branch response computation."""
import json,time,statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.scripts.split_mnist_usefulness import make
from bdh_cl.scripts.saved_config_response_retrieval import Selector,SOURCE,OUT
@torch.no_grad()
def main():
 torch.set_num_threads(2);model=make('local',gate='hard');model.load_state_dict(torch.load(SOURCE/'34_local_sequential.pt',weights_only=True));state=torch.load(OUT/'selectors/34.pt',weights_only=True);routes=state['routes'];router=Selector(torch.randn(10,5,10));router.load_state_dict(state['selector']);temp=state['temperature'];rows=[]
 for batch in [1,128]:
  x=torch.rand(batch,784)
  def automatic():return model(x).argmax(-1)
  def retrieved():
   b=torch.tanh(model.first(x)).reshape(-1,24,4);h=b[:,torch.arange(24)[None,:],routes];logits=F.linear(h,model.head.weight,model.head.bias);return router.prediction(logits,temp)[0]
  for name,fn in [('ordinary',automatic),('allroutes_selector',retrieved)]:
   for _ in range(30):fn()
   measurements=[]
   for _ in range(20):
    start=time.perf_counter()
    for _ in range(100):fn()
    measurements.append((time.perf_counter()-start)*1000/100)
   rows.append({'batch':batch,'method':name,'median_milliseconds':st.median(measurements)})
 (OUT/'timing.json').write_text(json.dumps({'protocol':'Seed34,2CPUthreads,torchno_grad,synthetic[0,1]inputs,samesharedfirstlayer.30warmups,20blocksx100calls,medianpercall. LatencydiagnosticnotGPU/FLOP/peakmemoryaudit.', 'rows':rows},indent=2));print(rows)
if __name__=='__main__':main()
