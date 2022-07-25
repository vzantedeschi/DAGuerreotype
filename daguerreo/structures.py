from abc import ABC

import torch

from .permutahedron import sparsemap_rank
from .utils import get_variances


class Structure(torch.nn.Module):

    @classmethod
    def initialize(cls, X, args): raise NotImplementedError()

    def forward(self):
        """
        This is for the unconditional computation of a probability distribution over complete DAG structures.

        Returns:
            Triplet => (probabilities, complete_dags, regularization_term)

        """

        raise NotImplementedError()


class _ScoreVectorStructure(Structure, ABC):

    def __init__(self, d, theta_init, l2_reg_strength, **kwargs) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(theta_init.unsqueeze(1), requires_grad=True)
        self.M = torch.triu(torch.ones((d, d)), diagonal=1)
        self.l2_reg_strength = l2_reg_strength

    @classmethod
    def initialize(cls, X, args, bilevel=False):
        d = X.shape[1]  # number of features= number of nodes

        # initialize the parameter score vector
        if args.smap_init_theta == "variances":
            theta_init = get_variances(X)
            m = theta_init.median()
            theta_init = (
                (theta_init - m).clone().detach()
            )
        else:
            theta_init = torch.zeros(d)
        return cls(d, theta_init, args.l2_theta, **cls._hps_from_args(args))

    @classmethod
    def _hps_from_args(cls, args): return {}

    def regularizer(self):
        return self.l2_reg_strength * (self.theta**2).sum()


class SparseMapSVStructure(_ScoreVectorStructure):
    """
    Class for learning the score vector with the sparseMAP operator.
    """

    def __init__(self, d, theta_init, l2_reg_strength, smap_tmp, smap_init, smap_iter) -> None:
        super().__init__(d, theta_init, l2_reg_strength)
        self.smap_tmp = smap_tmp
        self.smap_init = smap_init
        self.smap_iter = smap_iter

    @classmethod
    def _hps_from_args(cls, args):
        return {
            'smap_tmp': args.smap_tmp,
            'smap_init': args.smap_init,
            'smap_iter': args.smap_iter,
        }

    def forward(self):
        # call the sparseMAP rank procedure, it returns a vector of probabilities and one of sorted indices,
        # FIXME re-check if indieces are in ascending or descending order!!!!!!
        alphas, orderings = sparsemap_rank(
            self.theta / self.smap_tmp,  # the actual parameter, this is a good place to do perturb & map insted
            init=self.smap_init, max_iter=self.smap_iter)
        # TODO possibly add eval branch that returns the MAP here, although - not so sure why we should take the MAP,
        #  can't we keep the probability distribution also at eval time?
        return alphas, self.M[orderings[..., None], orderings[:, None]], self.regularizer()


class TopKSparseMaxSVStructure(_ScoreVectorStructure):
    # TODO
    pass


AVAILABLE = {
    'sp_map': SparseMapSVStructure,
    'tk_sp_max': TopKSparseMaxSVStructure
    # TODO add oracle fixed, etc, etc..
}

DEFAULT = 'sp_map'
