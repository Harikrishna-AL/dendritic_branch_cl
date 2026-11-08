"""Post-training CPU timing without profiler/tracemalloc overhead."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from bdh_cl.scripts.deep_predictive_memory_pilot import Model,stream

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);torch.set_num_threads(2)
    rows=json.loads((out/'results.json').read_text());result=[]
    with torch.no_grad():
        for r in rows:
            model=Model(r['arm']);model.load_state_dict(torch.load(out/f'{r["seed"]}_{r["order"][0]}_{r["arm"]}.pt',weights_only=True)[-1])
            x=next(stream(99,1,1))[0];times=[]
            for repeat in range(3):
                model.reset()
                for _ in range(30):_,q,_=model.predict(x);model.commit(q)
                start=time.perf_counter()
                for _ in range(300):_,q,_=model.predict(x);model.commit(q)
                times.append((time.perf_counter()-start)*1000/300)
            result.append({'seed':r['seed'],'order':r['order'],'arm':r['arm'],'median_ms':statistics.median(times),'repeats_ms':times})
    (out/'latency.json').write_text(json.dumps(result,indent=2))
    summary={a:statistics.mean(r['median_ms'] for r in result if r['arm']==a) for a in ('branch_memory','history_only','central_memory')}
    (out/'latency_summary.json').write_text(json.dumps({'milliseconds_per_16_stream_step':summary,'method':'CPU 2threads,3x300repeats afterwarmup,no profiler;same observation repeated;unoptimized toy'},indent=2));print(summary)
if __name__=='__main__':main()
