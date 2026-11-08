"""Presentation view: explicitly exclude class-mean controls, preserve raw data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
rows=json.loads((OUT/'all_baselines.json').read_text())
rows=[r for r in rows if r['kind'] not in ['centroid','offline']]
own=next(r for r in rows if r['label']=='Branch + auxiliary loss')
replay_free=[r for r in rows if r['kind'] in ['branch','other','rfe']]
blue='#126bb2'
gray='#a4adb6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(14,8.6))
grid=fig.add_gridspec(2,2,height_ratios=[3.2,1],width_ratios=[1,1.1],hspace=.48,wspace=.5)
ax=fig.add_subplot(grid[0,0])
ordered=sorted(replay_free,key=lambda r:r['accuracy'])
yy=np.arange(len(ordered))
ax.barh(yy,[r['accuracy'] for r in ordered],height=.62,
        color=[blue if r is own else gray for r in ordered])
ax.errorbar([r['accuracy'] for r in ordered],yy,xerr=[r['sd'] for r in ordered],fmt='none',color='#344250',capsize=3,lw=1)
for i,r in enumerate(ordered):
    ax.text(r['accuracy']+r['sd']+1.2,i,f"{r['accuracy']:.2f}%",va='center',fontsize=10,
            color=blue if r is own else '#46515b',fontweight='bold' if r is own else 'normal')
labels=[r['label'].replace('Branch + auxiliary loss','YOUR METHOD').replace(' (TMLR 2026), adapted',' 2026†').replace('Numenta (','Numenta (') for r in ordered]
ax.set(yticks=yy,yticklabels=labels,xlim=(0,90),xlabel='Final ten-class accuracy (%)',title='A. No stored historical examples')
ax.get_yticklabels()[-1].set_color(blue)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.grid(axis='x',alpha=.15)
ax.set_axisbelow(True)

bx=fig.add_subplot(grid[0,1])
replay_kinds=['replay','derpp','er_ace','ocar']
for r in rows:
    is_own=r is own
    is_replay=r['kind'] in replay_kinds
    color=blue if is_own else '#b28249' if is_replay else gray
    bx.errorbar(r['mb'],r['accuracy'],yerr=r['sd'],fmt='none',color=color,capsize=2,lw=1)
    bx.scatter(r['mb'],r['accuracy'],s=390 if is_own else 60,marker='*' if is_own else 's' if is_replay else 'o',
               color=color,edgecolor='white',zorder=5 if is_own else 3)
for label,offset in [('Branch + auxiliary loss',(38,-14)),('DER++: 1,000 images',(35,-12)),
                     ('OCAR (ICML 2025), adapted',(-95,-33)),('RFE (TMLR 2026), adapted',(12,-15))]:
    r=next(r for r in rows if r['label']==label)
    text='YOUR METHOD\n74.76% · 2.136 MB' if r is own else 'DER++ / 1,000\n94.52%' if r['kind']=='derpp' else 'OCAR 2025†\n93.06%' if r['kind']=='ocar' else 'RFE 2026†\n59.26%'
    bx.annotate(text,(r['mb'],r['accuracy']),xytext=offset,textcoords='offset points',fontsize=10,
                color=blue if r is own else '#665647',fontweight='bold' if r is own else 'normal',
                bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=1),
                arrowprops=dict(arrowstyle='-',color=blue if r is own else '#8d969e'))
bx.scatter([],[],marker='s',color='#b28249',label='Stores historical examples')
bx.scatter([],[],marker='o',color=gray,label='No historical examples')
bx.set(xscale='log',xlim=(1.85,24),ylim=(0,104),xlabel='Retained numeric state (MB, log scale)',
       ylabel='Final ten-class accuracy (%)',title='B. All measured neural methods')
bx.grid(alpha=.15)
bx.legend(loc='lower right',frameon=False,fontsize=9)

cx=fig.add_subplot(grid[1,:])
cx.hlines(0,74.76333280404408,97.39,color='#c18d3a',lw=5)
cx.scatter([74.76333280404408],[0],s=250,marker='*',color=blue,zorder=3)
cx.scatter([97.39],[0],s=100,facecolor='white',edgecolor='#a97723',linewidth=2,zorder=3)
cx.annotate('Autonomous: 74.76%',(74.76333280404408,0),xytext=(0,-25),textcoords='offset points',ha='center',color=blue,fontweight='bold')
cx.annotate('Oracle route: 97.39%',(97.39,0),xytext=(0,-25),textcoords='offset points',ha='center',color='#936516')
cx.text(86.08,.25,'22.63 percentage points to investigate',ha='center',fontweight='bold',color='#936516',fontsize=13)
cx.set(xlim=(68,104),ylim=(-.6,.8),yticks=[],xticks=[],title='C. Project target: make retained capabilities autonomously accessible')
for spine in cx.spines.values():spine.set_visible(False)
fig.suptitle('Dendritic routing: a concrete research opportunity',x=.04,y=.98,ha='left',fontsize=21,fontweight='bold')
fig.text(.04,.928,'Your method highlighted · class-mean methods excluded from this presentation view',fontsize=12,color='#53606c')
fig.subplots_adjust(left=.19,right=.97,top=.85,bottom=.16)
fig.text(.04,.075,'Panels A/B: measured autonomous performance. Panel C: privileged diagnostic, not an achievable-performance prediction.',fontsize=10,color='#53606c')
fig.text(.04,.045,'† Adapted recent methods. Error bars: SD across 3 seed means × 2 orders. Memory excludes optimizer and peak working RAM.',fontsize=10,color='#53606c')
fig.text(.04,.015,'The model leads this selected no-example-storage comparison; higher-accuracy replay methods remain visible in Panel B.',fontsize=10,color='#53606c')
for ext in ['png','svg']:fig.savefig(OUT/f'04_project_highlight_no_class_means.{ext}',dpi=190,facecolor='white')
plt.close(fig)
(OUT/'presentation_subset.json').write_text(json.dumps({'excluded':['raw-pixel class means','frozen-feature class means','offline joint reference'],
    'panel_a':'All measured neural methods with no stored historical examples; includes CE-only ablation',
    'panel_b':'All measured continual neural methods, including replay','data':rows},indent=2))
readme=OUT/'README.md'
text=readme.read_text()
addition=('\n## Highlighted presentation view\n\n[Project opportunity plot, class-mean methods excluded](04_project_highlight_no_class_means.png) '
         '(editable SVG also available). Panel A explicitly restricts comparison to no stored historical examples; '
         'Panel B retains all continual neural methods, including higher-accuracy replay methods. '
         'The blue star is visual emphasis, not a claim of being a statistical outlier. Original complete plots and data remain available.\n')
if '## Highlighted presentation view' not in text:readme.write_text(text+addition)
