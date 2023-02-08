import torch
import unittest

from ..permutahedron import sparsemax_rank, sparsemap_rank


def _invert_perm(pi):
    pi_inv = torch.zeros_like(pi)
    pi_inv[pi] = torch.arange(len(pi))
    return pi_inv


class ArgmaxCase(unittest.TestCase):

    def setUp(self):
        # generate 100 scores
        torch.manual_seed(42)
        self.x = torch.randn(100, 5)
        self.sort_perm = torch.argsort(self.x, dim=1)

    def test_correct_argmax_sparsemax(self):

        for i in range(len(self.x)):
            probas, perms = sparsemax_rank(self.x[i], max_k=2)
            assert torch.allclose(perms[0], _invert_perm(self.sort_perm[i]))

    def test_correct_argmax_sparsemap(self):

        for i in range(len(self.x)):
            probas, perms = sparsemap_rank(self.x[i], max_iter=2, init=False)
            assert torch.allclose(perms[0], _invert_perm(self.sort_perm[i]))


class PruneCase(unittest.TestCase):

    def test_sparsemax_prune(self):
        """Check that if scores are far enough apart output is smaller than k"""

        x = torch.tensor([1e6, 1e3, 1e0], dtype=torch.double)
        probas, perms = sparsemax_rank(x, max_k=10, prune_output=True)

        assert probas[0].item() == 1
        assert probas.shape[0] == 1
        assert perms.shape[0] == 1
