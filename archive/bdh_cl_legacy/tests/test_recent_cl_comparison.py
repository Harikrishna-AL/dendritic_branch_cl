"""Audit adapted inference and curvature against actual upstream functions."""
import ast
from copy import deepcopy
from types import SimpleNamespace

import einops
import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.scripts.recent_cl_comparison import RFE, Curvature, common, rfe_node
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC


def official_method(class_name,name):
    path = common.ROOT/'external/rfe/src/methods/rfe.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==class_name)
    node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope = dict(torch=torch,fn=F,ei=einops,ImageT=torch.Tensor,Output=SimpleNamespace)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


def test_rfe_all_class_prediction_matches_unmodified_official_inference():
    model = RFE(42,width=8)
    for i in range(5):
        model.aux.append(rfe_node('AuxFeatureExtractor')(32))
        if i<4:
            model.retro.append(rfe_node('Retrospector')(8,32,16))
    keys = [str(i) for i in range(5)]
    class Heads(nn.ModuleDict):
        def forward(self,x):
            return torch.cat([head(x) for head in self.values()],1)
    def chain(images,features,tkeys):
        for key in tkeys:
            i = int(key)
            features = model.retro[i](features,model.aux[i](images))
            yield key,features
    x = torch.randn(3,784)
    images = model.image(x)
    # Official extractor consumes images; use fixed corresponding main features.
    features = model.extractor(x)
    ref = SimpleNamespace(sync_device=lambda v:v,scale_fn=lambda v:v,extractor=lambda v:features,
                          tkey_to_retrospector={str(i):model.retro[i] for i in range(4)},
                          n_learned_target_classes=8,n_seen_target_classes=10,
                          tkey_to_classifier=Heads(dict(zip(keys,model.heads))),
                          tkey_to_d_output={key:2 for key in keys},chain_retrospect=chain)
    expected = official_method('RfeMethod','infer_logits')(ref,images).logits
    assert torch.allclose(model.scores(x),expected,atol=1e-7)
    assert torch.allclose(expected.sum(1),torch.ones(3))


def test_curvature_ema_uses_official_fim_and_carries_previous_information():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(3,4),nn.Tanh(),nn.Linear(4,10))
    x = torch.randn(6,3)
    y = torch.tensor([0,1,0,1,0,1])
    config = dict(lr=.001,ema=.2,tau_increment=.00001)
    tracker = Curvature(config)
    tracker.update(model,x,y,3)
    old = {key:tuple(t.clone() for t in pair) for key,pair in tracker.ema.data.items()}
    x2,y2 = torch.randn(6,3),torch.tensor([2,3,2,0,1,0])
    weights = torch.tensor([1.,1.,1.,2.,2.,2.])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x2,y2),batch_size=6)
    fresh = FIM(model,loader,PMatKFAC,n_output=10,device='cpu',lambda_=0.,weights=weights)
    tracker.update(model,x2,y2,3)
    for key,pair in tracker.ema.data.items():
        for observed,previous,new in zip(pair,old[key],fresh.data[key]):
            assert torch.allclose(observed,.8*previous+.2*new,atol=1e-6)
    assert tracker.known=={0,1,2,3}
    assert abs(tracker.tau-.00101)<1e-12
    assert all(not module._forward_hooks and not module._backward_hooks for module in model.modules())


def test_rfe_deployment_parameter_budget_includes_every_auxiliary_module():
    model = RFE(0)
    for i in range(5):
        model.aux.append(rfe_node('AuxFeatureExtractor')(32))
        if i<4:
            model.retro.append(rfe_node('Retrospector')(444,32,16))
    assert sum(p.numel() for p in model.parameters())==528014
    assert abs(528014-529210)/529210<.003
