import ast
from copy import deepcopy
import pickle
from types import SimpleNamespace

import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.scripts.leaderboard_missing_comparison import (
    PEC, GaussianStats, RidgeStats, MistLeaf, Mist, ROOT,
)


def upstream_node(class_name,method=None):
    path=ROOT/'external/pec/models/pec.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==class_name)
    if method:node=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==method)
    scope=dict(torch=torch,nn=nn,get_single_pec_network=lambda *a,**k:None)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[method or class_name]


def test_pec_matches_actual_official_observe_and_forward():
    torch.set_num_threads(2)
    ref=PEC(10);model=deepcopy(ref)
    Pair=upstream_node('PecStudentTeacherPair')
    pairs=[]
    for student in ref.students:
        pair=Pair.__new__(Pair);nn.Module.__init__(pair)
        pair.student,pair.teacher=student,ref.teacher
        pairs.append(pair)
    opts=[torch.optim.Adam(student.parameters(),lr=.001) for student in ref.students]
    wrapper=SimpleNamespace(cur_class_start=0,cur_class_end=2,pec_modules=pairs,opt=opts)
    x=torch.randn(8,784);y=torch.tensor([0,1,0,0,1,1,0,1])
    upstream_node('Pec','observe')(wrapper,x,y,x)
    own_opts=[torch.optim.Adam(student.parameters(),lr=.001) for student in model.students]
    target=model.teacher(x).detach()
    for c in [0,1]:
        own_opts[c].zero_grad()
        F.mse_loss(model.students[c](x[y==c]),target[y==c]).backward()
        norm=torch.nn.utils.clip_grad_norm_(model.students[c].parameters(),5.)
        assert norm<5 # Audit the shared-schedule clip is inactive on this step.
        own_opts[c].step()
    assert all(torch.allclose(a,b,atol=2e-7) for a,b in zip(ref.parameters(),model.parameters()))
    expected=upstream_node('Pec','forward')(wrapper,x)
    assert torch.allclose(model(x),expected,atol=1e-7)
    assert sum(p.numel() for p in model.parameters())==529145
    assert all(p.grad is None for p in model.teacher.parameters())


def test_gaussian_online_moments_and_qda_match_batch_density():
    rng=np.random.default_rng(12)
    x=rng.normal(size=(200,3));y=np.arange(200)%10
    model=GaussianStats(3)
    for start in range(0,len(x),7):model.update(x[start:start+7],y[start:start+7])
    for c in range(10):
        assert np.allclose(model.mean[c],x[y==c].mean(0))
        assert np.allclose(model.m2[c]/(model.count[c]-1),np.cov(x[y==c].T))
    shrink=.001;model.prepare('sqda',shrink)
    scores=[]
    for c in range(10):
        cov=(1-shrink)*np.cov(x[y==c].T)+shrink*np.eye(3)
        scores.append(multivariate_normal.logpdf(x[:15],mean=x[y==c].mean(0),cov=cov)+np.log(.1))
    assert np.array_equal(model.predict(x[:15]),np.stack(scores,1).argmax(1))


def test_rls_summaries_equal_sequential_sherman_morrison_solution():
    rng=np.random.default_rng(9);x=rng.normal(size=(30,4));y=np.arange(30)%10
    ridge=.7;model=RidgeStats(4)
    for i in range(0,30,5):model.update(x[i:i+5],y[i:i+5])
    model.prepare(ridge)
    inverse=np.eye(5)/ridge;weights=np.zeros((5,10))
    for row,label in zip(x,y):
        phi=np.r_[row,1.];gain=inverse@phi/(1+phi@inverse@phi)
        weights+=np.outer(gain,np.eye(10)[label]-phi@weights)
        inverse-=np.outer(gain,phi@inverse)
    assert np.allclose(model.head,weights,atol=1e-12)


def test_mist_inheritance_conserves_discounted_mass_and_gaussian_moments():
    leaf=MistLeaf(d=2)
    leaf.count[:2]=1000
    leaf.mean[:2]=np.array([[-.5,2.],[1.,-1.]])
    variance=np.ones((10,2))*2
    leaf.m2=variance*np.maximum(leaf.count[:,None]-1,1)
    children=leaf.split(0,.2,.6)
    for c in [0,1]:
        count=sum(child.count[c] for child in children)
        assert np.allclose(count,.6*leaf.count[c])
        mean=sum(child.count[c]*child.mean[c] for child in children)/count
        assert np.allclose(mean,leaf.mean[c])
        var=sum(child.count[c]*(child.variance()[c]+(child.mean[c]-mean)**2) for child in children)/count
        assert np.allclose(var,variance[c])
    assert all(not child.sketches for child in children)


def test_mist_checkpoint_roundtrip_preserves_sketches_and_predictions():
    rng=np.random.default_rng(31)
    model=Mist(dict(grace=200,k=64,delta=.1,tie=.05,alpha=.6))
    for i in range(300):model.update(rng.normal(size=50),i%2)
    probe=rng.normal(size=(5,50))
    restored=pickle.loads(pickle.dumps(model,protocol=5))
    assert model.nbytes()==restored.nbytes()
    for mode in ['mist_g','mist_k']:
        assert np.array_equal(model.predict(probe,mode),restored.predict(probe,mode))
