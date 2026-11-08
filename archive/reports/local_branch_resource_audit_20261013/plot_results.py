import json,statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
rows=json.loads((root/'results/local_graph_resource_benchmark_20261013/results.json').read_text())
models=['branch','central','sgd_mlp','adam_mlp','local_nb'];names=['Local branches','Central assemblies','MLP 256 / SGD','MLP 85 / Adam','Local class counts'];colors=['#177e89','#8055a2','#d17a22','#bd4553','#53785b']
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for col,ds in enumerate(['mnist','fashion_mnist']):
 for i,(arm,label,color) in enumerate(zip(models,names,colors)):
  rr=[[r for r in rows if r['dataset']==ds and r['domains']==d and r['arm']==arm] for d in [4,8]]
  mean=[100*st.mean(r['accuracy'] for r in rs) for rs in rr];err=[100*st.stdev(r['accuracy'] for r in rs) for rs in rr]
  axs[0,col].errorbar([4,8],mean,yerr=err,label=label,color=color,marker='o',capsize=3)
  axs[1,col].plot([4,8],[100*st.mean(r['forgetting'] for r in rs) for rs in rr],color=color,marker='o')
 axs[0,col].set_title('MNIST' if ds=='mnist' else 'Fashion MNIST');axs[0,col].set_ylabel('Final accuracy across all domains (%)');axs[0,col].set_ylim(45,100)
 axs[1,col].set_ylabel('Old-domain forgetting (percentage points)');axs[1,col].set_ylim(0,30)
 for ax in axs[:,col]:ax.set_xticks([4,8]);ax.set_xlabel('Sequential domains');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=9)
fig.suptitle('Equal data access: branches retain context, but the matched MLP is more accurate\nFresh paired seeds; accuracy bars show one sample standard deviation',fontsize=13)
fig.savefig(Path(__file__).parent/'accuracy_and_retention.png',dpi=180);fig.savefig(Path(__file__).parent/'accuracy_and_retention.svg');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
rs=[r for r in rows if r['dataset']=='mnist' and r['domains']==4]
x=np.arange(len(models))
for ax,key,scale,title in [(axs[0],'deployed_numeric_bytes',1e-6,'Stored model + memory (MB)'),(axs[1],'train_seconds',1,'Learning time (seconds)')]:
 vals=[st.mean(r[key] for r in rs if r['arm']==a)*scale for a in models];ax.bar(x,vals,color=colors)
 for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}' if key.endswith('bytes') else f'{v:.2f}',ha='center',va='bottom',fontsize=9)
 ax.set_title(title);ax.set_ylim(0,max(vals)*1.2)
vals=[st.mean(r['latency']['128']['median_ms'] for r in rs if r['arm']==a) for a in models]
axs[2].bar(x,vals,color=colors);axs[2].set_yscale('log');axs[2].set_title('128-example inference (ms; log scale)')
for i,v in enumerate(vals):axs[2].text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
for ax in axs:ax.set_xticks(x,names,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
fig.suptitle('Four-domain MNIST: storage parity does not establish compute parity\nTwo CPU threads; persistent numeric state excludes temporary working memory',fontsize=13)
fig.savefig(Path(__file__).parent/'resources.png',dpi=180);fig.savefig(Path(__file__).parent/'resources.svg')
