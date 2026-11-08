"""Native live tensor allocation peak, separate from Python memory and timing."""
import argparse,json
from pathlib import Path
import torch
from bdh_cl.scripts.deep_predictive_memory_pilot import Model,stream

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    torch.set_num_threads(2);report=[]
    for arm in ('branch_memory','history_only','central_memory'):
        model=Model(arm);model.load_state_dict(torch.load(out/f'0_0_{arm}.pt',weights_only=True)[-1]);x=next(stream(99,1,1))[0]
        with torch.no_grad():
            for _ in range(10):_,q,_=model.predict(x);model.commit(q)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True,with_stack=True) as prof:
                for _ in range(3):prediction,q,routes=model.predict(x);model.commit(q)
        path=out/f'tensor_memory_{arm}.json';prof.export_memory_timeline(str(path),device='cpu')
        timeline=json.loads(path.read_text());peak=max(sum(sizes) for sizes in timeline[1])
        report.append({'arm':arm,'peak_live_tracked_tensor_bytes':peak,
            'state_bytes':sum(b.numel()*b.element_size() for b in model.buffers()),
            'note':'CPU profiler live tensor timeline including tracked preexisting tensors;3inference steps;excludes Python/allocator/process overhead;not training/GPU peak'})
    (out/'tensor_memory.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
