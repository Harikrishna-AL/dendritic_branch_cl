"""Matched-total-step alternating-regime learning control, settings unchanged."""
import argparse,json,statistics
from pathlib import Path
import torch
from bdh_cl.scripts.usefulness_retention_synthetic import make,fit,audit
VARIANTS=('predict_winner','quality_soft','quality_all','uniform_all','shared','history','mlp','gru')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({'budget':'8192observations,balancedregimes inalternating128step blocks; singleAdam.003 cadence8, causal1step labels, no optimizer/fast resets','evaluation':'same256step heldout visits seed+5000; frozen slow weights; auditboth tasks','limits':'alternating training isjoint-exposure control,not anoracle optimalbound; changesfrequency comparedsequential4096 blocks; samefixedarchitecture/settings'},indent=2));records=[]
    for seed in (8,9,10):
        for first in (0,1):
            for v in VARIANTS:
                torch.manual_seed(seed);m=make(v);m.reset();opt=torch.optim.Adam(m.parameters(),lr=.003)
                fit(m,opt,seed+1000,(first,1-first)*32,128)
                phases=audit(m,seed+5000,(first,1-first)*2)
                r={'seed':seed,'first':first,'variant':v,'joint':phases};records.append(r);(out/'results.json').write_text(json.dumps(records,indent=2))
                print(seed,first,v,round(statistics.mean(p['all_mse'] for p in phases),3),flush=True)
if __name__=='__main__':main()
