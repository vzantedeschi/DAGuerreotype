import unittest
from daguerreo import sparsifiers as sp
import torch


class SparsifierCase(unittest.TestCase):

    def setUp(self, d=10, ns=12) -> None:
        self.sparsifier = sp.HardConcreteL0Sparsifier(1., d, ns)
        self.comp_dags = torch.randint(2, (ns, d, d)).float()

    def test_at_initialization(self):
        self.setUp()

        out, reg = self.sparsifier(self.comp_dags)
        assert torch.all(out >= 0)
        assert torch.all(out - self.comp_dags <= 0)

    def test_limit_negative(self):
        self.sparsifier.pi.data = -1000.*torch.ones_like(self.sparsifier.pi.data)
        out, reg = self.sparsifier(self.comp_dags)
        assert torch.allclose(out, torch.zeros_like(out))
        assert torch.allclose(reg, torch.zeros_like(reg))

    def test_limit_positive(self):
        self.sparsifier.pi.data = 1000. * torch.ones_like(self.sparsifier.pi.data)
        out, reg = self.sparsifier(self.comp_dags)
        assert torch.allclose(out, self.comp_dags)
        assert torch.allclose(reg, self.comp_dags.sum())
