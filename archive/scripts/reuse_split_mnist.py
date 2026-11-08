"""Numpy task-incremental Split MNIST reuse pilot, explicit pair identity.

14x14 inputs; trained tanh features; separate binary heads. Frozen growth adds
32 features per new task. Scratch retains independent 64-feature task models.
No replay, pretrained external encoder, or test-based model selection.
"""
import json
import gzip
import struct
from pathlib import Path
import numpy as np
SPLITS=((0,1),(2,3),(4,5),(6,7),(8,9))

def load_raw(directory):
    result={}
    for key,name in [('train_images','train-images-idx3-ubyte.gz'),('train_labels','train-labels-idx1-ubyte.gz'),
                     ('test_images','t10k-images-idx3-ubyte.gz'),('test_labels','t10k-labels-idx1-ubyte.gz')]:
        with gzip.open(Path(directory)/name,'rb') as f:raw=f.read()
        magic,count=struct.unpack('>II',raw[:8])
        if magic==2051:
            rows,cols=struct.unpack('>II',raw[8:16]);array=np.frombuffer(raw[16:],dtype=np.uint8).reshape(count,rows,cols)
        elif magic==2049:array=np.frombuffer(raw[8:],dtype=np.uint8).reshape(count)
        else:raise ValueError('Invalid IDX magic')
        result[key]=array
    return result

def downsample(images,side):
    n,h,w=images.shape;factor=h//side
    assert h==w and h%side==0
    return images.reshape(n,side,factor,side,factor).astype(np.float64).mean((2,4)).round().astype(np.uint8)

def features(x,w):return np.tanh(x@w)

def train(x,y,w,head,mask,rng):
    for ix in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/64))):
        h=features(x[ix],w);logits=h@head
        p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        p[np.arange(len(ix)),y[ix]]-=1;p/=len(ix)
        dw=x[ix].T@((p@head.T)*(1-h*h))
        dh=h.T@p
        w[:,mask]-=.15*dw[:,mask];head-=.15*dh

def accuracy(x,y,w,head):return float(np.mean((features(x,w)@head).argmax(1)==y))

def main():
    raw=load_raw('data_cache/mnist')
    x=downsample(raw['train_images'],14).reshape(-1,196).astype(np.float64)/255
    tx=downsample(raw['test_images'],14).reshape(-1,196).astype(np.float64)/255
    x=np.column_stack([x,np.ones(len(x),dtype=np.float64)])
    tx=np.column_stack([tx,np.ones(len(tx),dtype=np.float64)])
    runs=[]
    for seed in range(3):
        rng=np.random.default_rng(seed);data=[]
        for pair in SPLITS:
            ix=np.concatenate([rng.choice(np.flatnonzero(raw['train_labels']==digit),500,replace=False) for digit in pair])
            ti=np.flatnonzero(np.isin(raw['test_labels'],pair))
            data.append((x[ix],(raw['train_labels'][ix]==pair[1]).astype(int),tx[ti],(raw['test_labels'][ti]==pair[1]).astype(int)))
        initial=np.random.default_rng(seed+20).normal(0,.1,(197,64)).astype(np.float64)
        for arm in ['shared','frozen','frozen_grow','scratch','scratch_matched']:
            w=initial.copy();heads=[];models=[];matrix=[];curves=[]
            for task,(bx,by,ex,ey) in enumerate(data):
                if arm in ['scratch','scratch_matched']:
                    w=initial.copy()
                    if arm=='scratch_matched' and task:
                        extra=np.random.default_rng(seed+100+task).normal(0,.1,(197,32*task))
                        w=np.column_stack([w,extra])
                if arm=='frozen_grow' and task:
                    nw=np.random.default_rng(seed+100+task).normal(0,.1,(197,32)).astype(np.float64)
                    w=np.column_stack([w,nw])
                mask=np.ones(w.shape[1],dtype=bool)
                if arm in ['frozen','frozen_grow'] and task:
                    mask[:64 if arm=='frozen' else w.shape[1]-32]=False
                before=w[:,~mask].copy()
                head=np.zeros((w.shape[1],2),dtype=np.float64)
                stages=[accuracy(ex,ey,w,head)]
                for epoch in range(10):
                    train(bx,by,w,head,mask,np.random.default_rng(seed*1000+task*100+epoch))
                    stages.append(accuracy(ex,ey,w,head))
                assert np.array_equal(before,w[:,~mask])
                heads.append(head.copy());models.append(w.copy())
                values=[]
                for j in range(task+1):
                    ew=models[j] if arm in ['scratch','scratch_matched'] else w[:,:heads[j].shape[0]]
                    values.append(accuracy(data[j][2],data[j][3],ew,heads[j]))
                matrix.append(values);curves.append(stages)
            runs.append({'seed':seed,'arm':arm,'accuracy_matrix':matrix,'new_task_curves':curves,
                         'stored_feature_count':sum(m.shape[1] for m in models) if arm in ['scratch','scratch_matched'] else w.shape[1]})
            print(seed,arm,'final',matrix[-1],flush=True)
    out=Path('results/reuse_split_mnist');out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for arm in ['shared','frozen','frozen_grow','scratch','scratch_matched']:
        group=[r for r in runs if r['arm']==arm]
        summary.append({'arm':arm,'final_mean_accuracy':float(np.mean([r['accuracy_matrix'][-1] for r in group])),
            'mean_forgetting':float(np.mean([r['accuracy_matrix'][j][j]-r['accuracy_matrix'][-1][j] for r in group for j in range(4)])),
            'new_task_epoch1_accuracy':float(np.mean([r['new_task_curves'][j][1] for r in group for j in range(1,5)])),
            'stored_feature_count':group[0]['stored_feature_count']})
    (out/'results.json').write_text(json.dumps({'protocol':{'seeds':3,'train_per_digit':500,'epochs':10,'test':'all official test images','task_identity':'explicit','resolution':'14x14','replay':False},'runs':runs,'summary':summary},indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
