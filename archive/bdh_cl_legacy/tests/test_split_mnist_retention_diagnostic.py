import unittest
import torch

from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier
from bdh_cl.scripts.split_mnist_retention_diagnostic import features, ridge_fit, ridge_logits, pair_mask_score, score
from bdh_cl.scripts.split_mnist_route_audit import fixed_route_predictions


class Tests(unittest.TestCase):
    def test_saved_route_gather_agrees_with_independent_forward(self):
        torch.manual_seed(12)
        m = BranchClassifier('local', inputs=7, neurons=3, branches=4, gate='hard')
        x = torch.randn(20, 7)
        routes = torch.randint(0, 4, (5, 3))
        before = {k: v.clone() for k, v in m.state_dict().items()}
        for i in range(5):
            z = features(m, x, 'saved_routes', routes, torch.full((len(x),), i))
            expected = m.head(z).argmax(-1)
            actual = fixed_route_predictions(m, x, routes[i])
            self.assertTrue(torch.equal(actual, expected))
        for k, v in before.items():
            torch.testing.assert_close(m.state_dict()[k], v)

    def test_fixed_linear_probe_recovers_separable_classes(self):
        torch.manual_seed(13)
        y = torch.arange(10).repeat(20)
        z = torch.eye(10)[y] + torch.randn(len(y), 10) * .01
        fitted = ridge_fit(z, y)
        test = torch.eye(10) + torch.randn(10, 10) * .01
        self.assertTrue(torch.equal(ridge_logits(test, fitted).argmax(-1), torch.arange(10)))

    def test_pair_mask_is_privileged_and_different_from_global_score(self):
        y = torch.arange(10)
        logits = torch.eye(10)
        logits[:, 8:] += 100
        self.assertAlmostEqual(score(logits, y)['all_accuracy'], .2)
        self.assertAlmostEqual(pair_mask_score(logits, y)['all_accuracy'], 1)


if __name__ == '__main__':
    unittest.main()
