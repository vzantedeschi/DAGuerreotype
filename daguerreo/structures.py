from abc import ABC

import torch

from .permutahedron import sparsemap_rank, sparsemax_rank
from .utils import get_variances


class Structure(torch.nn.Module):

    @classmethod
    def initialize(cls, X, args): raise NotImplementedError()

    def forward(self):
        """
        This is for the unconditional computation of a probability distribution over complete DAG structures.
        When eval mode, returns the MAP (still as a triplet, with probability 1.)

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
        """
        Takes care of dimensionalities, initialization of the structure parameter theta
         (e.g. using nodes variances), and of any additional hyperparameters from args. Bilevel here is ignored as
         these objects have the same behaviour both in joint and bilevel optimization settings.
         """
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

    def forward(self):
        if self.training:
            return self._training_forward()
        else:
            map_ordering = self.map()
            return torch.ones(1), self.complete_graph_from_ordering(map_ordering), self.regularizer()

    def _training_forward(self):
        raise NotImplementedError()

    def map(self):
        """
        Returns: the most probable ordering (i.e. the mode of the distribution over ordering), that is
        inv_perm(ascending-argsort(theta).

        """
        # TODO not super happy of all these reshaping :; maybe we should keep the theta as a vector and reshape it
        #         when calling sparsemap  (btw @Vlad is there a specific reason why sparseMAP requires a matrix?)
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
                 smap_iter=100) -> None:
        super().__init__(d, theta_init, l2_reg_strength)

        self.smap_init = smap_init
        self.smap_iter = smap_iter

    @classmethod
    def _hps_from_args(cls, args):
        return {
            'smap_init': args.smap_init,
            'smap_iter': args.smap_iter,
        }

    def _training_forward(self, return_ordering=False):
        # call the sparseMAP rank procedure.
        # it returns a vector of probas and a matrix of integer permutations.
        # orderings[0] is the inverse permutation of argsort(self.theta)
        # (see tests)
        alphas, orderings = sparsemap_rank(
            # the actual parameter,
            # this is a good place to do perturb & map insted
            self.theta,
            init=self.smap_init, max_iter=self.smap_iter)
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
            # 'smax_tmp': args.smax_tmp,
            'smax_max_k': args.smax_max_k,
        }

    def _training_forward(self, return_ordering=False):
        # TODO @vlad here need to give a vector as input but sparsemap seem to require d x 1 matrix, right? make same?
        alphas, orderings = sparsemax_rank(self.theta.view(-1),
                                           max_k=self.smax_max_k)
        if return_ordering: return alphas, orderings
        return (alphas,
                self.complete_graph_from_ordering(orderings),
                self.regularizer())


AVAILABLE = {
    'tk_sp_max': TopKSparseMaxSVStructure,
    'sp_map': SparseMapSVStructure
    # TODO add oracle fixed, etc, etc..
}

DEFAULT = 'sp_map'
