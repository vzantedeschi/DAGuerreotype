from abc import ABC, abstractmethod

import torch
from torch.nn.modules.module import T

from .permutahedron import sparsemap_rank, sparsemax_rank
from .utils import get_variances


class Structure(torch.nn.Module):

    @classmethod
    def initialize(cls, X, args, *a): raise NotImplementedError()

    @abstractmethod
    def forward(self):
        """
        This is for the unconditional computation of a probability distribution over complete DAG structures.
        When eval mode, returns the MAP (still as a triplet, with probability 1.)

        Returns:
            Triplet => (probabilities, complete_dags, regularization_term)

        """

        raise NotImplementedError()

class FixedVectorStructure(Structure):

    def __init__(self, d, ordering, **kwargs) -> None:
        super().__init__()
        self.register_buffer('M', torch.triu(torch.ones((d, d)), diagonal=1))  # enables correct
        self.ordering = torch.argsort(ordering)

    @classmethod
    def initialize(cls, X, args, ordering):
        d = X.shape[1]  # number of features= number of nodes

        return cls(d, ordering, **cls._hps_from_args(args))

    @classmethod
    def _hps_from_args(cls, args): return {}

    def regularizer(self):
        return 0.

    def forward(self, return_ordering=False):
        if return_ordering:
            return torch.ones(1), self.ordering
        else:
            return torch.ones(1), self.M[self.ordering[:, None], self.ordering].unsqueeze(0), self.regularizer()

class _ScoreVectorStructure(Structure, ABC):

    def __init__(self, d, theta_init, l2_reg_strength, **kwargs) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(theta_init.unsqueeze(1), requires_grad=True)
        self.register_buffer('M', torch.triu(torch.ones((d, d)), diagonal=1))  # enables correct
        self.l2_reg_strength = l2_reg_strength

    @classmethod
    def initialize(cls, X, args, *a):
        """
        Takes care of dimensionalities, initialization of the structure parameter theta
         (e.g. using nodes variances), and of any additional hyperparameters from args. Bilevel here is ignored as
         these objects have the same behaviour both in joint and bilevel optimization settings.
         """
        d = X.shape[1]  # number of features= number of nodes

        # initialize the parameter score vector
        if args.init_theta == "variances":
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

    def forward(self):
        if self.training:
            return self._training_forward()
        else: # take MAP at inference
            map_ordering = self.map()
            return torch.ones(1), self.complete_graph_from_ordering(map_ordering), self.regularizer()

    @abstractmethod
    def _training_forward(self):
        raise NotImplementedError()

    def map(self):
        """
        Returns: the most probable ordering (i.e. the mode of the distribution over ordering), that is
        inv_perm(ascending-argsort(theta).

        """

        sorted_indices = torch.argsort(self.theta.view(-1), dim=0)
        inverse_permutation = torch.argsort(sorted_indices)
        return inverse_permutation.view(1, -1)

    def complete_graph_from_ordering(self, orderings):
        """

        Args:
            orderings: a tensor of orderings of dimensionality [num orderings, d]

        Returns: a tensor of dimensionality [num orderings, d, d] containing
                    the binary adjacency matrices of the complete dags corresponding to the input ordering

        """
        return self.M[orderings[..., None], orderings[:, None]]


class SparseMapSVStructure(_ScoreVectorStructure):
    """
    Class for learning the score vector with the sparseMAP operator.
    """

    def __init__(self, d, theta_init, l2_reg_strength=0., smap_init=False,
                 smap_iter_k=100) -> None:
        super().__init__(d, theta_init, l2_reg_strength)

        self.smap_init = smap_init
        self.smap_iter_k = smap_iter_k

    @classmethod
    def _hps_from_args(cls, args):
        return {
            'smap_init': args.smap_init,
            'smap_iter_k': args.smap_iter_k,
        }

    def _training_forward(self, return_ordering=False):
        # call the sparseMAP rank procedure.
        # it returns a vector of probas and a matrix of integer permutations.
        # orderings[0] is the inverse permutation of argsort(self.theta)
        # (see tests)
        alphas, orderings = sparsemap_rank(
            self.theta,
            init=self.smap_init, max_iter=self.smap_iter_k)
        if return_ordering: return alphas, orderings

        return (alphas,
                self.complete_graph_from_ordering(orderings),
                self.regularizer())


class TopKSparseMaxSVStructure(_ScoreVectorStructure):
    def __init__(self, d, theta_init, l2_reg_strength=0., smax_max_k=10) -> None:
        super().__init__(d, theta_init, l2_reg_strength)

        self.smax_max_k = smax_max_k

    @classmethod
    def _hps_from_args(cls, args):
        return {
            'smax_max_k': args.smax_max_k,
        }

    def _training_forward(self, return_ordering=False):
        alphas, orderings = sparsemax_rank(self.theta.view(-1),
                                           max_k=self.smax_max_k)
        if return_ordering: return alphas, orderings
        return (alphas,
                self.complete_graph_from_ordering(orderings),
                self.regularizer())


AVAILABLE = {
    'tk_sp_max': TopKSparseMaxSVStructure, # Top-K SparseMax
    'sp_map': SparseMapSVStructure, # SparseMAP
    "rnd_rank": FixedVectorStructure, # Fixed Random Ordering
    "true_rank": FixedVectorStructure, # Fixed True Ordering, Oracle structure
}

DEFAULT = 'sp_map'
