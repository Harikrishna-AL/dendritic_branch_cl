import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/dendritic_architecture_screen_20261105');r=json.loads((p/'results.json').read_text())
lines=['# Dendritic architecture screen','', 'One seed, two epochs per task; full training splits. All rows use image-only inference with all classes. Selection uses validation only. See [protocol](PROTOCOL.md).','', '| Dataset | Configuration | Parameters | State MB | Validation acc | Oracle acc | Forgetting pp | Training s |','|---|---|---:|---:|---:|---:|---:|---:|']
for a in r:
 lines.append(f"| {a['dataset']} | {a['name']} | {a['parameters']:,} | {a['inference_bytes']/1e6:.3f} | {100*a['validation']['accuracy']:.2f}% | {100*a['validation']['oracle_accuracy']:.2f}% | {100*a['forgetting']:.2f} | {a['training_seconds']:.1f} |")
lines+=['','## Validation-selected official test results','']
for a in r:
 if 'selected_test' in a:lines.append(f"- {a['dataset']}: {a['name']}, image-only test {100*a['selected_test']['accuracy']:.2f}%, oracle test {100*a['selected_test']['oracle_accuracy']:.2f}%.")
lines+=['','Memory is model tensor state including sparse indices; excludes optimizer, activations and diagnostic route archive. Training times include intermediate validation. Oracle is a diagnostic requiring true task identity. More branches are not assumed to help; equal-total-branch-unit controls distinguish allocation from capacity. This is an untuned pilot, not a conclusion about statistical significance or state of the art.']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
fig,axs=plt.subplots(1,2,figsize=(13,5))
for ax,d in zip(axs,['cifar10','cifar100']):
 for a in r:
  if a['dataset']!=d:continue
  ax.scatter(a['inference_bytes']/1e6,100*a['validation']['accuracy'],s=65)
  ax.annotate(a['name'],(a['inference_bytes']/1e6,100*a['validation']['accuracy']),xytext=(4,5),textcoords='offset points',fontsize=7)
 ax.set_title(d+' — image-only inference');ax.set_xlabel('Model state incl. indices (MB)');ax.set_ylabel('Final validation accuracy (%)');ax.grid(alpha=.2)
fig.suptitle('Branch allocation and compact sparse-input pilot (one seed)');fig.tight_layout();fig.savefig(p/'accuracy_memory.png',dpi=180);plt.close(fig)
print('\n'.join(lines))
