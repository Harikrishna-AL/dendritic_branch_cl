"""Exploratory retention around a previously observed natural electricity stress period."""
import argparse,copy,json,statistics
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts.branch_mechanism_ablations import MODES,make
from bdh_cl.scripts.electricity_usefulness_pilot import cohort,run
from bdh_cl.scripts.electricity_retention_pilot import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    d=cohort(0);y=d['year'];mo=d['month'];at=lambda year,month:int(np.flatnonzero((y==year)&(mo==month))[0])
    aend=at(2012,10);aheldend=at(2013,1);bstart=at(2014,7);bend=at(2014,8);bheldend=at(2014,9)
    (out/'source.py').write_text(Path(__file__).read_text());(out/'protocol.json').write_text(json.dumps({'period_selection':'exploratory: July/August2014 chosen from already observed high forecast errors; not untouched confirmation; original16customers,MT_007maydominate',
      'A':'train2012 after720h throughSeptember; olderheldoutOctober-December2012',
      'B':'trainJuly2014, forecastAugust2014heldout; same24h targets, rawmeasurements unchanged, no artificialtasktargettransform',
      'controls':'fixed eleven mechanismablations; seeds13-15; equaladditional744h moreA learningcontrol; optimizer/faststatepersist; coldheldoutauditsfreezeweights; qualityfeedbackdelayed24h',
      'limits':'notnatural contiguouslifetime becausehistorical heldout/gaps excluded; no taskIDinputs; exploratory same-dataset stresscase, not noveltyproof; reportper-client errors, no outlier removal; common_cue useslargerfaststate; parameter counts inrecords'},indent=2));records=[]
    for seed in (13,14,15):
        for mode in MODES:
            torch.manual_seed(seed);m=make(mode,25);m.reset();opt=torch.optim.Adam(m.parameters(),lr=.001)
            run(m,opt,d,720,aend,True,False);state=copy.deepcopy(m.state_dict());optstate=copy.deepcopy(opt.state_dict());recent=copy.deepcopy(m.recent)
            before=audit(m,d,aend,aheldend);newbefore=audit(m,d,bend,bheldend)
            run(m,opt,d,bstart,bend,True,False);after=audit(m,d,aend,aheldend);newafter=audit(m,d,bend,bheldend)
            torch.save({'afterA':state,'afterB':m.state_dict()},out/f'{seed}_{mode}.pt')
            m.load_state_dict(state);m.recent=copy.deepcopy(recent);opt.load_state_dict(optstate)
            run(m,opt,d,720,720+bend-bstart,True,False);moreA=audit(m,d,aend,aheldend)
            r={'seed':seed,'mode':mode,'parameters':sum(p.numel() for p in m.parameters()),'old_before':before['mse'],'old_after':after['mse'],'forgetting':after['mse']-before['mse'],'new_before':newbefore['mse'],'new_after':newafter['mse'],
               'new_gain':newbefore['mse']-newafter['mse'],'repeatA_old_change':moreA['mse']-before['mse'],'afterA_old':before,'afterB_old':after,'afterB_new':newafter}
            records.append(r);(out/'results.json').write_text(json.dumps(records,indent=2));print(seed,mode,'forget',round(r['forgetting'],4),'newgain',round(r['new_gain'],4),flush=True)
    lines=['# Exploratory natural electricity stress period','','July/August2014 period selected from earlier errors; settings fixed; three seeds; no signal/target transformations. Positive forgetting means worse older heldout performance.','','| Model | Old before | Old after | Forgetting | New before | New after | New gain | MoreA old change |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        rs=[r for r in records if r['mode']==mode];v=[statistics.mean(r[k] for r in rs) for k in ('old_before','old_after','forgetting','new_before','new_after','new_gain','repeatA_old_change')]
        lines.append('| '+mode+' | '+' | '.join(f'{a:.4f}' for a in v)+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
