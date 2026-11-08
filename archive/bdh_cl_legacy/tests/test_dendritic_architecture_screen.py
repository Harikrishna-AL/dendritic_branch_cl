import torch
from bdh_cl.scripts.dendritic_architecture_screen import Model


def test_counterfactual_is_actual_single_neuron_substitution():
    torch.manual_seed(9)
    model=Model(100,4,8,sparse=True)
    x=torch.randn(2,3,32,32)
    _,alternatives=model(x,counter=True)
    z=model.responses(x).tanh()
    route=model.error.argmin(-1)[None].expand(len(x),-1)
    h=z.gather(2,route[:,:,None]).squeeze(-1)
    h[:,2]=z[:,2,5]
    assert torch.allclose(alternatives[:,2,5],model.head(h),atol=1e-6)


def test_image_inference_requires_no_routes_and_does_not_mutate_memory():
    torch.manual_seed(10)
    model=Model(10,4,4,sparse=True)
    x=torch.randn(3,3,32,32)
    before={k:v.clone() for k,v in model.state_dict().items()}
    logits=model(x,'image')
    assert logits.shape==(3,10)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    logits.square().mean().backward()
    assert model.weight.grad.isfinite().all()
    assert model.encoder[0].weight.grad.abs().sum()>0
