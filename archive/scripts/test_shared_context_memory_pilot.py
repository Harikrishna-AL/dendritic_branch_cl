import unittest
import torch
from bdh_cl.scripts.shared_context_memory_pilot import Moments,Model

class Tests(unittest.TestCase):
    def test_full_moments_reconstruct_branch_traces(self):
        torch.manual_seed(7);m=Moments(3,streams=2);w=torch.randn(5,3);b=torch.randn(5)
        cross=torch.zeros(2,5);energy=torch.zeros(2,5);previous=None
        self.assertEqual(m.scores(w,b).abs().sum().item(),0)
        for _ in range(40):
            x=torch.randn(2,3);q=x@w.T+b
            if previous is not None:
                cross=.95*cross+.05*previous*q;energy=.95*energy+.05*previous.square()
            m.write(x);previous=q
            torch.testing.assert_close(m.scores(w,b),cross/(energy+.01),atol=2e-5,rtol=2e-5)
        m.reset();self.assertEqual(m.scores(w,b).abs().sum().item(),0)
    def test_capacity_and_causal_state(self):
        for v in ('baseline','centered','linear','softsign','neuron_shared','layer_shared'):
            m=Model(v);self.assertEqual(sum(p.numel() for p in m.parameters()),384)
            self.assertLessEqual(sum(b.numel() for b in m.buffers() if b.dtype==torch.float32),265*16)
            x=torch.randn(16,16);before=[b.clone() for b in m.buffers()]
            pred,q,_=m.predict(x)
            for a,b in zip(before,m.buffers()):torch.testing.assert_close(a,b)
            pred.square().mean().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
            m.commit(q);m.reset()
            self.assertTrue(all(not b.any().item() for b in m.buffers()))

if __name__=='__main__':unittest.main()
