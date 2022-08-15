# from ..structures import SparseMapSVStructure
import itertools

from interval import interval

from daguerreo import structures as sr

import unittest
import torch


def all_diff(a, eps):
    mem = list()
    for k, v in enumerate(a.view(-1)):
        int_v = interval((v.item() - eps, v.item() + eps))
        for s, ivl in enumerate(mem):
            if int_v & ivl:
                print('Intervals of vector entries intersect; discarding')
                return False
        mem.append(int_v)
    return True


class StructureCase(unittest.TestCase):

    def setUp(self, d=100, k=10) -> None:
        self.d = d
        init = torch.randn(self.d)
        self.smap_str = sr.SparseMapSVStructure(self.d, init)
        self.smax_str = sr.TopKSparseMaxSVStructure(self.d, init, k)

    def test_map(self):
        for i in range(100):
            self.setUp()
            alphas, complete_dags, reg = self.smap_str()
            map_ordering = self.smap_str.map()

            mat_map = self.smap_str.complete_graph_from_ordering(map_ordering)

            test = torch.allclose(complete_dags[0], mat_map[0])
            assert test

    def test_smax(self):
        d = 6
        for i in range(100):
            self.setUp(d)
            if all_diff(self.smap_str.theta, 1.e-5):
                alphas, orderings = self.smax_str._training_forward(True)
                weights = (orderings.float() @ self.smax_str.theta).ravel()

                all_weights = []
                for e in itertools.permutations(range(d)):
                    all_weights.append(torch.tensor(e).float() @ self.smax_str.theta)
                all_weights, _ = torch.tensor(all_weights).sort(descending=True)
                assert torch.allclose(all_weights[:len(weights)], weights)

