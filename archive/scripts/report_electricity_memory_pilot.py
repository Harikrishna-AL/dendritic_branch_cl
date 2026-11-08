"""Report real-data pilot, runtime memory erasure, and chronological errors."""
import json,statistics,copy,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/bdh_electricity_matplotlib')
import numpy as np
import torch
from bdh_cl.scripts.electricity_memory_pilot import Forecast,run,prepare,summarize

OUT=Path('results/electricity_memory_pilot_20261011')
class Erased(Forecast):
    def forecast(self,x):
        for c in self.context:c.reset()
        return super().forecast(x)

def main():
    torch.set_num_threads(2);rows=json.loads((OUT/'results.json').read_text());assert len(rows)==21
    baselines=json.loads((OUT/'baselines.json').read_text());data=prepare('data/electricity/electricity.zip','data/electricity/hourly16.pt')
    y=data['y'];years=data['year'];months=data['month'];start=int(np.flatnonzero(years==2013)[0]);end=len(y)-24
    erase=[]
    for r in rows:
        if r['arm'] not in ('branch_tanh','branch_linear'):continue
        m=Erased(r['arm']);m.reset();saved=torch.load(OUT/f'{r["seed"]}_{r["arm"]}_2012.pt',weights_only=False);m.load_state_dict(saved['model'])
        rs,_=run(m,None,y,years,months,start,end,False);erase.append({'seed':r['seed'],'arm':r['arm'],'frozen_erased':summarize(rs)})
    (OUT/'erasure.json').write_text(json.dumps(erase,indent=2))
    stats={}
    for arm in sorted({r['arm'] for r in rows}):
        rs=[r for r in rows if r['arm']==arm]
        stats[arm]={mode:{'mean_mse':statistics.mean(r[mode]['2014']['mse'] for r in rs),
                            'per_seed_mse':[r[mode]['2014']['mse'] for r in rs],
                            'mean_mae':statistics.mean(r[mode]['2014']['mae'] for r in rs)} for mode in ('frozen','online')}
    (OUT/'aggregate.json').write_text(json.dumps(stats,indent=2))
    lines=['# Electricity pilot interpretation','','24-hour forecasts;16 training-prefix-selected clients; three initialization seeds; errors standardized by first-month2012 client scales. The24h targets arrive before training updates, never before predictions.','','| Model | 2014 frozen MSE | 2014 online MSE |','|---|---:|---:|']
    for label,r in baselines.items():lines.append(f'| {label} | {r["2014"]["mse"]:.3f} | {r["2014"]["mse"]:.3f} |')
    for arm,s in stats.items():lines.append(f'| {arm} | {s["frozen"]["mean_mse"]:.3f} | {s["online"]["mean_mse"]:.3f} |')
    lines+=['','## Runtime erasure (weights frozen)','','| Branch model | Live MSE | Erased MSE |','|---|---:|---:|']
    for arm in ('branch_tanh','branch_linear'):
        value=statistics.mean(r['frozen_erased']['2014']['mse'] for r in erase if r['arm']==arm)
        lines.append(f'| {arm} | {stats[arm]["frozen"]["mean_mse"]:.3f} | {value:.3f} |')
    lines+=['','Erasure forces default first-branch routes in a model trained with memory-selected routes. It diagnoses dependence on runtime routing; it does not prove superiority to a model trained without memory.','', 'This is a real-data forecasting transfer test, not a demonstration of task separation or reduced forgetting. Client states are tracked in known parallel lanes; no customer/task labels are model inputs. Initial training, memory and optimizer initialization are shared consistently, but effective runtime memory budgets differ. Shared contexts store full covariance matrices and use much more state. GRU uses a detached hidden state (one-step truncated gradients) and the same observed25-feature window; this is a compact online control, not a tuned state-of-the-art GRU.','', 'No hyperparameters were selected from2013/2014 in this pilot. Frozen2013 is an untouched diagnostic validation period; online2013 supplies past delayed labels for adaptation before2014. There is one-pass training and no replay beyond the24h pending-label queue. Full-dataset generalization, stronger tuned baselines, multi-horizon results and controlled retention tests remain untested.']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,mode in zip(axes,('frozen','online')):
        for arm,col in [('branch_tanh','#3977ac'),('branch_linear','#27906c'),('shared','#b07936'),('history','#a35483'),('mlp','#555555'),('gru','#ed7054')]:
            rs=[r for r in rows if r['arm']==arm];values=[statistics.mean(r[mode]['2014']['monthly_mse'][str(m)] for r in rs) for m in range(1,13)]
            ax.plot(range(1,13),values,label=arm,color=col,linewidth=1.4)
        ax.plot(range(1,13),[baselines['daily']['2014']['monthly_mse'][str(m)] for m in range(1,13)],label='daily persistence',color='black',linestyle=':')
        ax.set_title('Frozen weights' if mode=='frozen' else 'Online:24h delayed labels');ax.set_xlabel('Month2014');ax.set_ylabel('Standardized MSE');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Electricity transfer:16 clients, three seeds,24h horizon')
    fig.savefig(OUT/'monthly.png',dpi=160);fig.savefig(OUT/'monthly.pdf');plt.close(fig)
    lines+=['',f'![Monthly errors]({(OUT/"monthly.png").resolve()})']
    (OUT/'interpretation.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:23]))
if __name__=='__main__':main()
