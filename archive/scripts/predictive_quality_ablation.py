"""Predictive target/maturation ablation and an actual naive deep baseline."""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.predictive_branches_mnist import Branches,MASK
from bdh_cl.scripts.predictive_router_ablation import embed
from bdh_cl.scripts.reuse_split_mnist import load_raw,downsample,SPLITS

class Predictors(Branches):
    def __init__(self,seed,target,maturity):
        super().__init__(seed,'frozen_reuse',.03)
        self.target_kind=target;self.maturity=maturity

    def responses(self,images):
        x,_,z,predicted,_,_=super().responses(images)
        if self.target_kind=='pixels':target=images[:,~MASK]
        else:
            im=images.reshape(-1,14,14)
            contrast=im-np.concatenate([np.zeros((len(im),14,1)),im[:,:,:-1]],axis=2)
            target=contrast.reshape(-1,196)[:,~MASK]
        weight=1+4*np.abs(target)
        errors=np.sum((predicted-target[:,None])**2*weight[:,None],axis=2)/weight.sum(1)[:,None]
        errors[:,~self.active]=np.inf
        return x,target,z,predicted,weight,errors

    def learn_pixels(self,images):
        x,target,z,predicted,weight,errors=self.responses(images)
        chosen=errors.argmin(1);best=errors.min(1)
        if self.frozen[self.active].all() and np.median(best)>.03 and not self.active.all():
            b=int(np.flatnonzero(~self.active)[0]);self.active[b]=True
            self.events.append({'step':self.step,'branch':b});chosen[best>.03]=b
        frozen=self.frozen.copy();old=self.encoder[frozen].copy();olddec=self.decoder[frozen].copy()
        for b in np.unique(chosen):
            ix=chosen==b
            if self.frozen[b]:continue
            err=2*(predicted[ix,b]-target[ix])*weight[ix]/weight[ix].sum(1)[:,None]
            de=z[ix,b].T@err/ix.sum();dz=(err@self.decoder[b].T)*(1-z[ix,b]**2)
            self.encoder[b]-=.2*x[ix].T@dz/ix.sum();self.decoder[b]-=.2*de
            self.age[b]+=1
            if self.age[b]>=self.maturity:self.frozen[b]=True
        assert np.array_equal(old,self.encoder[frozen]) and np.array_equal(olddec,self.decoder[frozen])
        self.step+=1

class Deep:
    def __init__(self,seed):
        rng=np.random.default_rng(seed+20)
        self.w1=rng.normal(0,.1,(197,64));self.w2=rng.normal(0,.1,(65,32));self.w3=np.zeros((33,10))
    def forward(self,x):
        x=np.column_stack([x,np.ones(len(x))]);h=np.tanh(x@self.w1)
        hb=np.column_stack([h,np.ones(len(x))]);z=np.tanh(hb@self.w2)
        zb=np.column_stack([z,np.ones(len(x))]);return x,h,hb,z,zb,zb@self.w3
    def observe(self,x,y):
        x,h,hb,z,zb,l=self.forward(x)
        p=np.exp(l-l.max(1,keepdims=True));p/=p.sum(1,keepdims=True);p[np.arange(len(y)),y]-=1;p/=len(y)
        g3=zb.T@p;dz=(p@self.w3[:-1].T)*(1-z*z);g2=hb.T@dz
        dh=(dz@self.w2[:-1].T)*(1-h*h);g1=x.T@dh
        self.w1-=.1*g1;self.w2-=.1*g2;self.w3-=.1*g3
    def predict(self,x):return self.forward(x)[-1].argmax(1)

def main():
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(float)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(float)/255
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed)
        data=[np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==d),500,replace=False) for d in pair]) for pair in SPLITS]
        d=Deep(seed);curve=[];pairs=[]
        for stage,ix in enumerate(data):
            for epoch in range(10):
                order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                for b in np.array_split(order,16):d.observe(x[b],raw['train_labels'][b])
            p=d.predict(tx);seen=raw['test_labels']<2*(stage+1)
            curve.append(float(np.mean(p[seen]==raw['test_labels'][seen])))
            pairs.append([float(np.mean(p[np.isin(raw['test_labels'],pair)]==raw['test_labels'][np.isin(raw['test_labels'],pair)])) for pair in SPLITS[:stage+1]])
        runs.append({'seed':seed,'target':'none','maturity':0,'arm':'naive_deep','accuracy':curve,'pairs':pairs})
        print(seed,'naive_deep',curve,flush=True)
        for target_kind in ['pixels','contrasts']:
            for maturity in [20,200]:
                m=Predictors(seed,target_kind,maturity);random_encoder=m.encoder.copy()
                router=np.zeros((99,8));router[-1]=.15
                arms=['reconstruction','frozen_router','ungated','random_active']
                proto={a:np.zeros((10,64)) for a in arms};count=np.zeros(10,int)
                logs={a:{'seed':seed,'target':target_kind,'maturity':maturity,'arm':a,'accuracy':[],'pairs':[]} for a in arms}
                for stage,ix in enumerate(data):
                    oldclasses=np.flatnonzero(count);oldp={a:proto[a].copy() for a in arms}
                    for epoch in range(10):
                        order=np.random.default_rng(seed*1000+stage*100+epoch).permutation(ix)
                        for b in np.array_split(order,16):
                            frozen=m.frozen.copy();m.learn_pixels(x[b]);v,_,_,_,_,errors=m.responses(x[b])
                            eligible=m.active&~frozen
                            router[:,eligible]-=.005*v.T@(v@router[:,eligible]-errors[:,eligible])/len(b)
                            h={a:embed(m,x[b],a,router,random_encoder)[0] for a in arms}
                            labels=raw['train_labels'][b]
                            for c in np.unique(labels):
                                chosen=labels==c;n=int(chosen.sum());total=count[c]+n
                                for a in arms:proto[a][c]=(count[c]*proto[a][c]+h[a][chosen].sum(0))/total
                                count[c]=total
                    for a in arms:
                        assert np.array_equal(proto[a][oldclasses],oldp[a][oldclasses])
                        h=embed(m,tx,a,router,random_encoder)[0];classes=np.flatnonzero(count)
                        distance=((h[:,None]-proto[a][classes])**2).sum(2);p=classes[distance.argmin(1)]
                        seen=raw['test_labels']<2*(stage+1);logs[a]['accuracy'].append(float(np.mean(p[seen]==raw['test_labels'][seen])))
                        logs[a]['pairs'].append([float(np.mean(p[np.isin(raw['test_labels'],pair)]==raw['test_labels'][np.isin(raw['test_labels'],pair)])) for pair in SPLITS[:stage+1]])
                for a in arms:
                    logs[a]['active']=int(m.active.sum());logs[a]['events']=m.events;runs.append(logs[a])
                print(seed,target_kind,maturity,{a:logs[a]['accuracy'][-1] for a in arms},'active',int(m.active.sum()),flush=True)
    keys=sorted({(r['target'],r['maturity'],r['arm']) for r in runs});summary=[]
    for key in keys:
        group=[r for r in runs if (r['target'],r['maturity'],r['arm'])==key];values=[r['accuracy'][-1] for r in group]
        summary.append({'target':key[0],'maturity':key[1],'arm':key[2],'accuracy':float(np.mean(values)),
                        'sd':float(np.std(values,ddof=1)),'mean_active':float(np.mean([r.get('active',0) for r in group]))})
    out=Path('results/predictive_quality_ablation');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'runs':runs,'summary':summary},indent=2))
    lines=['# Predictive feature quality ablation','','Three paired seeds; same sequential five pairs, 500 training examples per digit, ten epochs, all official test images, 14x14 inputs. No task ID, replay or pair-based test masking. Prototype arms compete over all seen classes. Deep baseline predicts all ten digits throughout.','','Targets: complementary pixels or complementary horizontal pixel contrasts (current pixel minus left neighbor). Both are available from the current image without digit labels. Trainable branch encoders/decoders freeze after 20 or 200 assigned minibatch updates. Other predictor hyperparameters fixed. Routing arms share identical predictors and active capacity within each target/maturity/seed condition. Random-active control uses identical slot activation but fixed random encoders. Changing maturity also changes recruitment and which data each branch sees; this is not a pure training-duration isolation.','','Naive deep baseline: tanh MLP 196→64→32→10 with bias, SGD 0.1, cross entropy, current examples only. Prototype branches use 98 input pixels, eight 8-dimensional slots, pixel loss and separate class means. Thus objectives, architecture, inference candidate set and capacity differ. This is a system-level pilot comparison, not a matched causal claim that dendrites beat deep networks. No test-based model or hyperparameter selection.','','| Target | Maturity | Arm | Final accuracy | SD | Active branches |','|---|---|---|---|---|---|']
    for s in summary:lines.append(f"| {s['target']} | {s['maturity']} | {s['arm']} | {100*s['accuracy']:.2f}% | {100*s['sd']:.2f} | {s['mean_active']:.2f} |")
    lines+=['','Frozen predictor and old prototype invariance assertions passed during each run. All configurations, seed-level stage curves, final pair results and recruitment events are in results.json. Grid results are exploratory; best-setting claims require fresh seeds and more data. Prediction targets are hand-designed and branch allocation/freezing are engineered. Learned prototype means include immature representations.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Saved',len(runs),'runs',flush=True)

if __name__=='__main__':main()
