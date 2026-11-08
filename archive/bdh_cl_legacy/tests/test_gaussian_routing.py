import dataclasses
import torch
from bdh_cl.config import BDHSpec
from bdh_cl.models.bdh_wrapper import BDHWrapper
from bdh_cl.models.gaussian_routed_bdh import OnlineRouter,RoutedBDH


def test_native_sum_equivalence():
    spec=BDHSpec(n_layer=2,n_embd=16,n_head=2,mlp_internal_dim_multiplier=4,dropout=0,neuron_unit='dendritic',neuron_unit_kwargs={'n_branches':4,'soma':'sum'})
    torch.manual_seed(1);native=BDHWrapper(spec)
    routed=RoutedBDH(spec,'sum',side=7);routed.load_state_dict(native.state_dict())
    x=torch.randint(0,256,(2,8))
    torch.testing.assert_close(native(x)[0],routed(x),rtol=0,atol=0)


def test_router_rejection_and_read_only():
    torch.manual_seed(2);r=OnlineRouter(16,2)
    images=torch.randint(0,256,(128,4,4))
    r.fit(images,'gaussian');before={k:v.clone() for k,v in r.state_dict().items()}
    gate,assignment,cache=r.route(images,'gaussian')
    assert (gate[:,:,0]==1).all()
    assert (gate.sum(-1)<=2).all()
    assert (assignment==0).float().mean()<.1
    for k,v in before.items():assert torch.equal(v,r.state_dict()[k])
    old=r.count.clone();r.update(cache,'gaussian');assert (r.count>=old).all()
    r.threshold.fill_(1e10)
    gate,_,cache=r.route(images,'gaussian');old=r.count.clone()
    r.update(cache,'gaussian')
    assert gate[:,:,1:].sum()==0
    torch.testing.assert_close(r.count,old)


def test_statistics_and_spatial_sensitivity():
    r=OnlineRouter(16,2)
    im=(16*torch.arange(16)).reshape(1,4,4)
    assert not torch.equal(r.project(im),r.project(im.flip(-1)))
    r.count.fill_(2);r.total.fill_(6);r.square.fill_(20);r.refresh()
    torch.testing.assert_close(r.mean,torch.full_like(r.mean,3))
    torch.testing.assert_close(r.var,torch.ones_like(r.var))


def test_specialist_gradient_mask():
    spec=BDHSpec(n_layer=1,n_embd=16,n_head=2,mlp_internal_dim_multiplier=4,dropout=0,neuron_unit='dendritic',neuron_unit_kwargs={'n_branches':4,'soma':'sum'})
    m=RoutedBDH(spec,'gaussian',side=7)
    im=torch.randint(0,256,(32,7,7));m.router.fit(im,'gaussian')
    m.router.threshold.fill_(1e10)
    m(torch.randint(0,256,(2,8)),im[:2]).square().mean().backward()
    for unit in (m.neuron_unit_x,m.neuron_unit_y):
        g=unit.weight.grad.reshape(2,16,4,-1)
        assert g[:,:,1:].abs().sum()==0
        assert g[:,:,0].abs().sum()>0
