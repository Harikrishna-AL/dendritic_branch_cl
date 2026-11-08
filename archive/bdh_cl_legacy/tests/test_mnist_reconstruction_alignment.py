import torch
from torch.nn import functional as F
from bdh_cl.scripts.mnist_reconstruction_alignment import views,rec_error,gate_step,decoder_objective
from bdh_cl.scripts.mnist_masked_gate_refinement import DecoderModel


def test_outer_classification_gradient_reaches_decoder_through_gate_step():
 torch.manual_seed(101);m=DecoderModel(101);x=torch.randn(4,784);y=torch.arange(4)
 raw,z,zm,mask,target=views(m,x,torch.Generator().manual_seed(9));s=raw.abs().clone().requires_grad_();r=rec_error(m,zm,s,target,mask);s1=gate_step(r,s,True)
 logits=F.linear((z*s1.softmax(-1)).sum(-1),m.head.weight.detach(),m.head.bias.detach());loss=F.cross_entropy(logits,y);loss.backward()
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.decoder.parameters())
 assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('decoder.'))
 assert all(torch.isfinite(p.grad).all() for p in m.decoder.parameters() if p.grad is not None)


def test_new_losses_do_not_backpropagate_into_proven_core_and_masks_match():
 states=[]
 for arm in ['decoder_only','ranking','meta','ranking_meta']:
  torch.manual_seed(101);m=DecoderModel(101);x=torch.randn(3,784);y=torch.arange(3);gen=torch.Generator().manual_seed(123)
  loss,_=decoder_objective(m,x,y,arm,gen);loss.backward();states.append(gen.get_state())
  assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('decoder.'))
 assert all(torch.equal(states[0],s) for s in states)
