import torch
from bdh_cl.scripts.mnist_masked_gate_refinement import DecoderModel,refine,masked_loss


def test_masked_loss_ignores_visible_targets():
 pred=torch.tensor([[.2,.5,.7]]);target=torch.tensor([[.1,.3,.9]]);mask=torch.tensor([[True,False,True]])
 altered=target.clone();altered[:,1]=100
 assert torch.equal(masked_loss(pred,target,mask),masked_loss(pred,altered,mask))


def test_refinement_preserves_model_and_initial_classification():
 torch.manual_seed(101);m=DecoderModel(101);m.requires_grad_(False);x=torch.randn(3,784);before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():expected=m.prediction(x,mode='magnitude_soft').argmax(-1)
 result=refine(m,x,steps=1)
 assert torch.equal(result[0]['prediction'],expected)
 assert torch.isfinite(result[1]['reconstruction']).all()
 assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
