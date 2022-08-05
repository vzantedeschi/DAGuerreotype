# from ..structures import SparseMapSVStructure
from daguerreo import structures as sr

import unittest
import torch


class StructureCase(unittest.TestCase):

    def setUp(self, d=100) -> None:
        self.d = d
        init = torch.randn(self.d)
        self.smap_str = sr.SparseMapSVStructure(self.d, init)

    def test_map(self):
        for i in range(100):
            self.setUp()
            alphas, complete_dags, reg = self.smap_str()
            map_ordering = self.smap_str.map()

            mat_map = self.smap_str.complete_graph_from_ordering(map_ordering)

            test = torch.all(complete_dags[0] == mat_map[0])
            assert test
