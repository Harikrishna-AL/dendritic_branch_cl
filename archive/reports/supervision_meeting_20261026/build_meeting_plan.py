"""Meeting handout and editable source; no new experimental claims."""
from pathlib import Path
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
PDF=ROOT/'output/pdf/Harikrishna_FOSTER_PRO_Research_Plan.pdf'
PDF.parent.mkdir(parents=True,exist_ok=True)
INK=colors.HexColor('#17333D');TEAL=colors.HexColor('#247B80');GRAY=colors.HexColor('#53646C');LIGHT=colors.HexColor('#EFF5F5')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=21,leading=24,textColor=INK,spaceAfter=9))
styles.add(ParagraphStyle(name='Sub2',fontName='Helvetica',fontSize=9,leading=12,textColor=GRAY,spaceAfter=10))
styles.add(ParagraphStyle(name='Head2',fontName='Helvetica-Bold',fontSize=11.2,leading=14,textColor=TEAL,spaceBefore=10,spaceAfter=5))
styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=9.5,leading=13.2,textColor=INK,spaceAfter=6))
styles.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=8,leading=10.5,textColor=GRAY,spaceAfter=4))
styles.add(ParagraphStyle(name='Cell2',fontName='Helvetica',fontSize=8.4,leading=11.2,textColor=INK))
styles.add(ParagraphStyle(name='CellHead2',fontName='Helvetica-Bold',fontSize=8.4,leading=11.2,textColor=colors.white))
story=[];md=[]
def para(text,small=False):
 story.append(Paragraph(text,styles['Small2' if small else 'Body2']))
 md.append(text.replace('<b>','**').replace('</b>','**').replace('<br/>','\n')+'\n')
def heading(text):story.append(Paragraph(text,styles['Head2']));md.append('## '+text+'\n')
def title(number,text,subtitle):
 if number>1:story.append(PageBreak());md.append('\n---\n')
 story.append(Paragraph(f'{number:02d} / MEETING DISCUSSION',styles['Sub2']))
 story.append(Paragraph(text,styles['Title2']));md.append('# '+text+'\n')
 story.append(Paragraph(subtitle,styles['Sub2']));md.append(subtitle+'\n')
def box(text):
 p=Paragraph(text,styles['Body2']);t=Table([[p]],colWidths=[505])
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#C9DDDE')),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.append(t);story.append(Spacer(1,7));md.append('> '+text.replace('<b>','**').replace('</b>','**')+'\n')
def table(headers,rows,widths):
 data=[[Paragraph(escape(x),styles['CellHead2']) for x in headers]]+[[Paragraph(x,styles['Cell2']) for x in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),TEAL),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#CDDDDF'))]))
 story.append(t);story.append(Spacer(1,6))
 md.append('| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |')
 for row in rows:md.append('| '+' | '.join(x.replace('<br/>','; ').replace('<b>','**').replace('</b>','**') for x in row)+' |')
 md.append('')

title(1,'Two projects, one research direction','Harikrishna Pillai | Discussion with Subhendu | 26 October 2026 | Proposal for agreement')
box('<b>Meeting objective:</b> agree the remaining FOSTER deliverables, a distinct 15-credit semester research question, supervision, and a written allocation of paid versus assessed work.')
heading('The common scientific question')
para('Can separating computation and short-lived state across dendritic branches help a network learn new situations without disrupting earlier skills? The current evidence motivates this question, but does not establish a uniquely dendritic continual-learning advantage.')
table(['Project','Purpose','Scope boundary'],[
 ['<b>FOSTER</b><br/>Paid grant; reported 6 h/week over 4 months','Implement and evaluate the original short-term branch-memory hypothesis.','Baseline implementation, controlled evaluation and honest documentation of positive and negative findings.'],
 ['<b>PRO research project</b><br/>15 credits; one complete semester','Investigate a new mechanism of selective learning beyond the agreed funded baseline.','Design and causally test branch-local plasticity; develop an independent scientific contribution and paper-oriented report.']
],[105,195,205])
heading('What the experiments currently establish')
table(['Evidence','Result','Interpretation'],[
 ['Fading activity memory [E1]','Synthetic MSE: competition only 0.725; branch memory + competition 0.761. Lower is better.','This tested short-term rule did not improve the control. It does not refute every branch-memory rule.'],
 ['Historical configuration [E2]','Split MNIST: 19.40% automatic; 98.02% with historical gates and unchanged final weights/head.','Old skills were accessible under assisted configurations. True digit-pair selection and stored gates were required.'],
 ['Automatic retrieval [E3]','MNIST: local cues 76.39%; + neighbors 77.48%; direct ridge 85.61%.','A small local-communication effect; extra memory is required. No corresponding gain on Fashion MNIST.'],
 ['Input-driven branches [E4]','Joint: 96.30%; sequential: 16.26%. Matched MLP sequential: 20.14%.','End-to-end Adam training works jointly; sequential acquisition is sometimes unstable and retention is not improved.']
],[110,195,200])
para('<b>Important:</b> these rows describe different models/protocols, not a single improvement curve. Fading state, cumulative input prototypes, optimizer moments and temporary gates are different forms of state. A retrieval gain from prototypes is not proof that fading branch memory reduces forgetting.',small=True)
para('Evidence IDs: E1 = fading-state pilot (11 Oct); E2 = configuration/readout diagnostic (11 Oct); E3 = locked retrieval studies (24 Oct); E4 = input-driven inhibition study (26 Oct). Full local records are indexed in the editable plan.',small=True)
para('<b>Proposed direction:</b> stop expanding inference-time routing searches. First separate stable learning from retention, then test whether branch-local control of plasticity has an advantage over equally sparse ordinary networks.')

title(2,'FOSTER: complete the funded hypothesis','Working title: Learning without forgetting - testing short-term memory in neural networks')
heading('Aim and minimum scientific deliverable')
para('Evaluate whether fading branch state changes specialization, adaptation and interference relative to long-term weights alone. A well-controlled negative result is a legitimate scientific outcome; whether the proposed deliverables satisfy the grant must be agreed with the supervisors.')
box('<b>Keep the original hypothesis explicit:</b> short-term state must actually evolve during the stream and fade. Input prototypes or saved configurations may be diagnostic extensions, but cannot substitute for the grant-core fading-memory test.')
table(['Work package','Deliverable and evidence'],[
 ['F1. Audit and define the baseline','Document the slow weights, fading state, update timing and available feedback. Verify read-before-write, no task-ID/boundary input, and what persists during evaluation. Freeze a reproducible baseline.'],
 ['F2. Consolidate the core comparisons','State on/off and competition on/off; branch-specific versus matched shared state; ordinary-network controls. Use existing synthetic prediction and chronological electricity experiments, plus Split MNIST diagnostics where relevant. Rerun only unresolved checks.'],
 ['F3. Explain outcomes and costs','Report acquisition, adaptation after switches and retention separately. Count trainable parameters, fast state, auxiliary storage and runtime. Mark any oracle routing or old-label readout probe as assisted analysis.'],
 ['F4. Handover and dissemination','Short technical report, a small set of clear figures, runnable commands and experiment index. Record supported claims, limitations and which new research questions remain open.']
],[115,390])
heading('Provisional closure effort - subject to remaining paid hours')
table(['Task','Illustrative remaining effort'],[
 ['Baseline/provenance audit','6 hours'],['Only essential control checks','8 hours'],['Results and figures','6 hours'],['Report and handover','4 hours'],['Total','24 hours = 4 weeks at 6 hours/week, only if available in the grant balance.']
],[300,205])
para('The original reported commitment is four months at six hours per week. This is not a request to restart or extend that period. Actual funded dates, work already charged and remaining hours are still to be checked. Reduce or reorder closure tasks if less time remains.')
heading('Acceptance and scope to agree')
para('Request agreement on a concrete completion checklist rather than a positive accuracy target. Confirm whether the compact benchmark suite is sufficient under the approved proposal; larger CIFAR/CORe50 commitments should be explicitly prioritized or revised, not silently dropped. Do not treat the current research history as automatic proof that FOSTER is complete.')
para('The new selective-weight-update mechanism is proposed for PRO only if the supervisors agree it lies beyond the funded scope; otherwise the task split or research question must change.',small=True)

title(3,'PRO: selective learning in branched networks','Proposed 15-credit semester project | A scientific extension beyond the agreed FOSTER baseline')
box('<b>Primary research question:</b> Can a branch-local plasticity mechanism preserve input-selective computations during task-ID-free continual learning better than an ordinary network with comparable sparsity, memory and compute?')
heading('Proposed contribution')
para('Develop and test one bounded family of branch-local learning controls, rather than add another configuration retriever. Short-lived activity/error state would regulate <b>where learning is written</b>, alongside slower weights. The exact rule should be selected after a focused literature review. A scalar activity-based slowdown already failed to show a specific benefit, so it remains a negative control rather than the proposed novelty.')
para('Use global backpropagation as the initial teaching baseline. Replacing all credit assignment with a biologically local rule is outside the minimum project. Biological inspiration is a design motivation; any plausibility claim must specify the actual signals available to each branch.')
heading('Three linked experiments')
table(['Stage','Question and discriminating comparison'],[
 ['P1. Establish stable learning','Compare Adam and momentum-free SGD with separately selected learning rates; check initialization/normalization of the soma. Match joint and current-task learning before interpreting forgetting. Keep all failed runs.'],
 ['P2. Test selective plasticity','Choose at most two rule candidates. Compare branch-local state with shared state, shuffled assignments and state off. Include an equally sparse/gated ordinary network. Measure old-loss change at a matched amount of new learning.'],
 ['P3. Test transfer and access','Confirm locked settings on fresh seeds and both orders. Measure retention and automatic inference; use probes or historical configurations only as explicitly assisted diagnostics of what remains accessible.']
],[105,400])
heading('Benchmarks and controls')
para('<b>Mechanism:</b> a controlled sequential feature-combination task with adjustable overlap. <b>Primary image setting:</b> task-ID-free domain-incremental MNIST/Fashion MNIST, keeping the output labels fixed to reduce new-class readout confounds. <b>Secondary transfer:</b> strict Split MNIST and, if feasible, Split Fashion MNIST with all ten outputs eligible. Report domain- and class-incremental claims separately.')
para('Use cached data and small models initially. A larger image benchmark or shared pretrained encoder is optional after the mechanism passes; any pretraining exposure must be declared and identical for controls. Frozen/random-feature and direct-classifier references remain important wherever comparable.')
heading('Decision criteria before scaling')
para('<b>Learning gate:</b> target acquisition within two percentage points of the matched control on validation data. <b>Mechanism gate:</b> a retention/interference benefit over shared, shuffled and ordinary sparse controls, not merely over unrestricted Adam. <b>Confirmation:</b> lock settings before five fresh seeds and both orders; give paired seed-level intervals and at least one transfer condition. These are proposed criteria for agreement, not achieved results.')
para('<b>Outputs:</b> a new rule/protocol, causal ablations, full resource accounting, reproducible code and a paper-oriented report. A precise limitation or negative mechanism result is the fallback scientific outcome; publication acceptance is not a promised deliverable.')

title(4,'Semester milestones and agreement','Dates, module requirements and the task allocation require supervisor confirmation')
table(['Core weeks','Milestone / review decision'],[
 ['1-2','Literature map, baseline handover, hypotheses and frozen evaluation protocol. Agree what is new PRO work.'],
 ['3-5','Stable acquisition: optimizer and soma controls. Go/no-go before interpreting retention.'],
 ['6-9','At most two plasticity candidates; local/shared/shuffled and sparse-network pilots. Stop uninformative sweeps.'],
 ['10-12','Locked fresh-seed/order confirmation and one dataset/setting transfer.'],
 ['13-15','Causal analysis, parameter/state/compute accounting and stress tests.'],
 ['16 onward','Draft course report and manuscript; use the remaining semester for feedback, revisions and any justified follow-up. Formal module deadline takes precedence.']
],[75,430])
heading('Proposed task boundary - not a retroactive allocation')
table(['Work','Allocation to confirm'],[
 ['Existing implementation/pilot history','Background with recorded provenance. Confirm actual funded tasks/hours; do not simply relabel already paid work as assessed work.'],
 ['Baseline audit, funded comparisons, FOSTER report','FOSTER work packages F1-F4, within agreed paid hours.'],
 ['New rule design and independent causal study','PRO work packages P1-P3, after or alongside FOSTER only with explicit task separation.'],
 ['One eventual paper using both sets of findings','Possible shared scientific output; identify each contribution and funding acknowledgement. Paid work is not submitted again as new assessed work.']
],[190,315])
heading('Decisions requested in this meeting')
para('1. Will Subhendu supervise the 15-credit project, and what are its report, presentation and submission requirements?<br/>2. What remains necessary to finish FOSTER, and how many paid hours remain?<br/>3. Is selective plasticity the right PRO question, or should the earlier retrieval-focused proposal be retained instead? Choose one primary direction.<br/>4. Which tasks, start/transition date and separate work logs document the allocation?<br/>5. Agree a regular check-in, available compute and a paper target after the first mechanism review.')
heading('Selected starting literature')
para('Poirazi &amp; Papoutsi (2020), dendritic computation: <link href="https://doi.org/10.1038/s41583-020-0301-7" color="#247B80">doi:10.1038/s41583-020-0301-7</link><br/>Legenstein &amp; Maass (2011), branch-specific plasticity and binding: <link href="https://doi.org/10.1523/JNEUROSCI.5684-10.2011" color="#247B80">doi:10.1523/JNEUROSCI.5684-10.2011</link><br/>Cichon &amp; Gan (2015), branch-specific plasticity/interference: <link href="https://doi.org/10.1038/nature14251" color="#247B80">doi:10.1038/nature14251</link><br/>Ashley et al. (2021), optimizer effects on forgetting: <link href="https://arxiv.org/abs/2102.07686" color="#247B80">arXiv:2102.07686</link><br/>Chavlis &amp; Poirazi (2024), parameter-efficient dendritic networks: <link href="https://arxiv.org/abs/2404.03708" color="#247B80">arXiv:2404.03708</link>',small=True)

def footer(canvas,doc):
 canvas.setStrokeColor(TEAL);canvas.setLineWidth(.6);canvas.line(45,36,A4[0]-45,36)
 canvas.setFillColor(GRAY);canvas.setFont('Helvetica',7.2)
 canvas.drawString(45,24,'Harikrishna Pillai | Discussion draft | FOSTER / PRO allocation subject to agreement')
 canvas.drawRightString(A4[0]-45,24,str(doc.page))

doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=40,bottomMargin=47,title='FOSTER and PRO Research Plan - Harikrishna Pillai',author='Harikrishna Pillai')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
text='\n'.join(md)
import re
text=re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>',r'[\2](\1)',text)
text=text.replace('&amp;','&')
text+='''

## Evidence index for the meeting

E1 - Grant-core fading-state pilot: [BRANCH_SHORT_MEMORY_EXPERIMENTS.md](../../BRANCH_SHORT_MEMORY_EXPERIMENTS.md).

E2 - Historical configuration and readout diagnostics: [summary](../../results/split_mnist_retention_diagnostic_20261011/summary.md). This also reports that an MLP frozen-feature readout can recover 82.15% using all old training labels; random features reach 83.43%. Do not claim that an ordinary MLP necessarily lost all information.

E3 - Confirmed automatic retrieval and costs: [24 October report](../overnight_branch_research_20261024/MORNING_REPORT.md). Separate cohorts support the low-memory cue gain (59.69% to 66.29%) and the small communication gain (76.39% to 77.48%). They are not a single paired improvement.

E4 - Direct input-driven inhibition with Adam: [26 October report](../input_driven_inhibition_20261026/REPORT.md).

Other bounded negative directions: [recurrent gate updater](../recurrent_branch_gate_20261026/REPORT.md), [energy-ranking pilot](../branch_energy_landscape_20261026/REPORT.md), [activity-regulation controls](../branch_binding_interference_20261025/REPORT.md).

These local reports remain the evidence records. No claim of new independent replication is made by this planning document.
'''
(BASE/'Research_plan.md').write_text(text)
print(PDF)
