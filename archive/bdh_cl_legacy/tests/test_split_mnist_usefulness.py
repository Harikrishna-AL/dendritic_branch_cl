import unittest

import torch
from torch.nn import functional as F

from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier, make, evaluate


class Tests(unittest.TestCase):
    def test_counterfactual_is_actual_single_neuron_replacement(self):
        torch.manual_seed(5)
        m = BranchClassifier('local', inputs=7, neurons=3, branches=4)
        x = torch.randn(6, 7)
        _, candidates = m(x, counterfactual=True)
        branches = torch.tanh(m.first(x).reshape(-1, 3, 4))
        h = (branches * m.gates()).sum(-1)
        for neuron in range(3):
            for branch in range(4):
                replaced = h.clone()
                replaced[:, neuron] = branches[:, neuron, branch]
                torch.testing.assert_close(candidates[:, neuron, branch], m.head(replaced))

    def test_forward_is_order_independent_and_evaluation_does_not_learn(self):
        m = make('local')
        x = torch.randn(20, 784)
        y = torch.arange(10).repeat(2)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        order = torch.randperm(len(x))
        torch.testing.assert_close(m(x)[order], m(x[order]))
        evaluate(m, x, y, list(range(10)))
        evaluate(m, x, y.flip(0), list(range(10)))
        for k, v in before.items():
            torch.testing.assert_close(m.state_dict()[k], v)

    def test_shared_memory_pools_neurons_and_budget_matches(self):
        m = make('shared')
        m.observe(torch.rand(24, 4))
        torch.testing.assert_close(m.error, m.error[:1].expand_as(m.error))
        count = lambda model: sum(p.numel() for p in model.parameters())
        self.assertLess(abs(count(m) - count(make('mlp'))) / count(m), .001)
        for arm in ('local', 'shared', 'uniform'):
            self.assertEqual(count(make(arm)), count(m))

    def test_auxiliary_objective_trains_every_branch(self):
        m = make('local')
        x, y = torch.randn(16, 784), torch.arange(16) % 10
        _, candidate = m(x, counterfactual=True)
        target = y[:, None, None].expand(candidate.shape[:-1]).reshape(-1)
        F.cross_entropy(candidate.reshape(-1, 10), target).backward()
        gradients = m.first.weight.grad.reshape(24, 4, 784).norm(dim=-1)
        self.assertTrue((gradients > 0).all().item())

    def test_hard_competition_chooses_exactly_one_branch_per_neuron(self):
        m = make('local', gate='hard')
        scores = torch.ones_like(m.error)
        chosen = torch.arange(m.neurons) % m.branches
        scores[torch.arange(m.neurons), chosen] = -10
        m.observe(scores)
        self.assertTrue(torch.equal(m.gates().argmax(-1), chosen))
        torch.testing.assert_close(m.gates().sum(-1), torch.ones(m.neurons))
        self.assertTrue(((m.gates() == 0) | (m.gates() == 1)).all().item())


if __name__ == '__main__':
    unittest.main()
