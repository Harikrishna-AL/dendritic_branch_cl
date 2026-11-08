import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
reports=[json.loads(f.read_text()) for f in sorted(root.glob('seed*.json'))]
summary={}
for arm in reports[0]['arms']:
    rows=[]
    for report in reports:
        rec=report['arms'][arm];st=rec['stages'];last=st[-1]
        if len(st)!=5:continue
        time=sum(s['train_seconds'] for s in st)
        base=sum(s['train_seconds'] for s in report['arms']['sum']['stages'])
        row={'final_accuracy':last['seen_accuracy'],'old_accuracy':last['old_accuracy'],'new_accuracy':last['new_accuracy'],
             'forgetting':float(np.mean([max(s['accuracy_by_pair'][t] for s in st[t:])-last['accuracy_by_pair'][t] for t in range(4)])),
             'train_seconds':time,'overhead_percent':100*(time/base-1),'parameters':rec['parameters'],'router_bytes':rec['router_bytes']}
        if last['routing']:
            row['old_route_agreement']=float(np.mean([s['agreement_with_acquisition'] for s in last['routing'][:4]]))
            row['final_rejection']=float(np.mean([s['rejected_fraction'] for s in last['routing']]))
        rows.append(row)
    summary[arm]={k:{'mean':float(np.mean([r[k] for r in rows])),'std':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else 0} for k in rows[0]} if rows else {}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Gaussian routing: Split-MNIST results','',f'{len(reports)} seeds; variability is sample standard deviation, not confidence interval. See PROTOCOL.md for constraints.','', '| Arm | Final accuracy | Old classes | New classes | Forgetting | Time vs sum |','|---|---:|---:|---:|---:|---:|']
for arm,stats in summary.items():
    def fmt(key,scale=100):return f"{stats[key]['mean']*scale:.2f} ± {stats[key]['std']*scale:.2f}"
    lines.append(f"| {arm} | {fmt('final_accuracy')}% | {fmt('old_accuracy')}% | {fmt('new_accuracy')}% | {fmt('forgetting')} pp | {fmt('overhead_percent',1)}% |")
lines+=['','## Routing','']
for arm,stats in summary.items():
    if 'old_route_agreement' in stats:lines.append(f"- {arm}: old-input route agreement {100*stats['old_route_agreement']['mean']:.1f}%; final rejection {100*stats['final_rejection']['mean']:.1f}%.")
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
