from abc import ABC
from itertools import chain

import torch.nn
from sklearn.linear_model import LinearRegression, LassoLarsIC
import numpy as np
from torch import nn

import utils
from sparsifiers import Sparsifier


class Equations(torch.nn.Module):

    def forward(self, X, dags):
        """

        Args:
            X: the datapoints
            dags: tensor of dags (already "sparsified", possibly :) )

        Returns:
            The triplet [reconstructed X (one per dag), dag (possibly sparsified), regularizer)

        """
        raise NotImplementedError()

    def fit(self, X, complete_dags, dag_sparsifier: Sparsifier, loss):
        self.train()
        dag_sparsifier.train()

    @classmethod
    def initialize(cls, X, args, joint=False):
        raise NotImplementedError()


class LARSAlgorithm(Equations):

    def __init__(self, d) -> None:
        super().__init__()
        self.d = d
        self.W = None

    def forward(self, X, dags):
        # this only works if fit is done, we should check this with an
        # assert not self.training
        return torch.einsum("np,opc->onc", X, self.W), self.W != 0, 0.

    def fit(self, X, dags, dag_sparsifier=None, loss=None):
        super(LARSAlgorithm, self).fit(X, dags, dag_sparsifier, loss)

        LR = LinearRegression(normalize=False, n_jobs=1)
        LL = LassoLarsIC(criterion="bic", normalize=False)

        x_numpy = X.detach().numpy()
        masks_numpy = dags.long().detach().numpy()

        self.W = np.zeros((len(masks_numpy), self.d, self.d))

        for m, mask in enumerate(masks_numpy):
            for target in range(self.d):

                covariates = np.nonzero(mask[:, target])[0]

                if len(covariates) > 0:  # if target is not a root node

                    LR.fit(x_numpy[:, covariates], x_numpy[:, target].ravel())
                    weight = np.abs(LR.coef_)

                    LL.fit(x_numpy[:, covariates] * weight, x_numpy[:, target].ravel())
                    self.W[m, covariates, target] = LL.coef_ * weight

            assert (self.W[m, mask == 0] == 0).all(), (self.W[m], mask)  # FIXME why this?

        self.W = torch.from_numpy(self.W)

    @classmethod
    def initialize(cls, X, args, joint=False):
        return cls(X.shape[1])


def masked_x(X, dags):
    """
    Args:
        X: the datapoints
        dags: 3-d tensor of dags

    Returns:
        Z, a masked expanded version of X. Z is a 4-d tensor where:
            - the first dimension indexes the dag
            - the second indexes the example (indexed with s below)
            - the third and fourth dimension represent the example expanded so that
                    Z^s_ij = 0          if j is not parent of i
                           = X_sj       if j is parent of i!
    """
    return torch.einsum("opc,np->oncp", dags, X)


class ParametricGDFitting(Equations, ABC):

    def __init__(self, d, num_structures, l2_reg_strength, optimizer, n_iters) -> None:
        super().__init__()
        self.l2_reg_strength = l2_reg_strength
        self.d = d
        self.num_structures = num_structures
        self.optimizer = optimizer
        self.n_iters = n_iters
        if self.num_structures: self.init_parameters(self.num_structures)

    def init_parameters(self, num_structures):
        raise NotImplementedError()

    @classmethod
    def initialize(cls, X, args, joint=False):
        return cls(X.shape[1], 1 if joint else None,
                   **cls._hps_from_args(args))

    @classmethod   # TODO ideally this should be the other way around... args from hp!
    def _hps_from_args(cls, args):
        return {
            # 'optimizer': lambda _vrs: utils.get_optimizer(_vrs, name=args.optimizer, lr=args.lr),
            'optimizer': lambda _vrs: utils.get_optimizer(_vrs, name='sgd', lr=0.1),
            # 'n_iters': args.num_inner_iters,
            'n_iters': 100,
            'l2_reg_strength': args.l2_eq
        }

    def fit(self, X, complete_dags, dag_sparsifier, loss):
        super(ParametricGDFitting, self).fit(X, complete_dags, dag_sparsifier, loss)
        n_dags = len(complete_dags)
        dag_sparsifier.init_parameters(n_dags)
        self.init_parameters(n_dags)

        inner_vars = chain(self.parameters(), dag_sparsifier.parameters())
        # todo also here we'd better decouple the hyperparameters
        inner_opt = self.optimizer(inner_vars)

        for inner_iters in range(self.n_iters):
            inner_opt.zero_grad()

            dags, sparsifier_reg = dag_sparsifier(complete_dags)
            x_hat, dags, equations_reg = self(X, dags)

            inner_objective = loss(x_hat, X, dim=(1, 2)) + sparsifier_reg + equations_reg
            inner_objective.sum().backward()

            if inner_iters % 10 == 0:
                pass

            inner_opt.step()


    def regularizer(self):
        """
        l2 regularizer on the weights of the parametric model, one per dag_structure

        Returns: [num_structures] vector of l2 regularizers (already mulitplied by the coefficient).

        """
        return self.l2_reg_strength * torch.stack([
                (p**2).sum(tuple(range(1, p.ndim))) for p in self.parameters()
                # todo probably better mean than sum here
            ]).sum(0)


class LinearEquations(ParametricGDFitting):

    def __init__(self, d, num_structures, l2_reg_strength, optimizer, n_iters) -> None:
        self.W = None
        super().__init__(d, num_structures, l2_reg_strength, optimizer, n_iters)

    def init_parameters(self, num_structures):
        self.W = nn.Parameter(
            torch.zeros(num_structures, self.d, self.d), requires_grad=True
        )  # W[:, p, c] one weight from parent p to child c
        # W[0]'s column c reconstructs node c
        # TODO why no bias?

    def forward(self, X, dags):
        # reconstruct the child from the parents, one per dag!
        x1 = torch.einsum("oncp,opc->onc", masked_x(X, dags), self.W)
        return x1, dags, self.regularizer()


class NonlinearEquations(Equations):
    # todo
    pass


# ---- previous stuff // commented out because i still need to implement this  -----

#
# class NonLinearEquationsOld(EquationsOld):
#     def __init__(
#         self, d, num_equations=1, hidden=2, activation=torch.nn.functional.leaky_relu
#     ):
#
#         super(NonLinearEquationsOld, self).__init__(d)
#
#         self.num_equations = num_equations
#         self.hidden = hidden
#
#         self.W = nn.Parameter(
#             torch.randn(num_equations, d, d, hidden)
#             * 0.05,  # TODO: check for better value of std
#             requires_grad=True,
#         )
#
#         self.bias = nn.Parameter(
#             torch.zeros(num_equations, 1, d, hidden), requires_grad=True
#         )
#
#         self.activation = activation
#
#         self.final_map = nn.Parameter(
#             torch.randn(num_equations, d, hidden) * 0.05, requires_grad=True
#         )
#
#     def forward(self, masked_x):
#
#         out = torch.einsum("oncp,opch->onch", masked_x, self.W)  #
#
#         out = self.activation(out + self.bias)
#
#         return torch.einsum("onch,och->onc", out, self.final_map)
#
#     def l2(self):
#
#         out = torch.sum(self.W**2, dim=(-3, -2, -1))  # one l2 per set of equations
#         out += torch.sum(self.bias**2, dim=(-3, -2, -1))
#         out += torch.sum(self.final_map**2, dim=(-2, -1))
#
#         return out
#
#
#
# class NNL0Estimator(Estimator, nn.Module):
#     def __init__(
#         self,
#         d,
#         num_structures,
#         bernouilli_init=0.5,
#         linear=True,
#         hidden=1,
#         activation=torch.nn.functional.leaky_relu,
#     ):
#
#         nn.Module.__init__(self)
#         # TODO: rename to sparsity
#         self.structure = BernoulliStructure(
#             d, num_structures, initial_value=bernouilli_init
#         )
#
#         if linear:
#             self.equations = LinearEquationsOld(d, num_structures)
#         else:
#             self.equations = NonLinearEquationsOld(
#                 d, num_structures, hidden=hidden, activation=activation
#             )
#
#     def forward(self, x, maskings):
#
#         dags = self.structure(maskings)
#
#         masked_x = torch.einsum(
#             "opc,np->oncp", dags, x
#         )  # for each ordering (o), data point (i) and node (c): vector v, with v_p = x_ip if p is potentially a parent of c, 0 otherwise
#
#         x_hat = self.equations(masked_x)
#
#         return x_hat
#
#     def fit(self, x, maskings, loss, args):
#
#         self.train()
#
#         optimizer = get_optimizer(
#             self.parameters(), name=args.optimizer, lr=args.lr
#         )  # to update structure and equations
#
#         # inner loop
#         for inner_iters in range(args.num_inner_iters):
#
#             x_hat = self(x, maskings)  # (num_structures, n, d)
#
#             objective = loss(x_hat, x, dim=(-2, -1))
#             objective += args.pruning_reg * self.pruning() + args.l2_reg * self.l2()
#
#             objective.sum().backward()
#
#             optimizer.step()
#             optimizer.zero_grad()
#
#             self.project()
#
#         self.eval()
#
#     def project(self):
#         self.structure.B.project()
#
#     def pruning(self):
#         return self.structure.l0()
#
#     def l2(self):
#         return self.equations.l2()
#
#     def get_structure(self, masking):
#         return self.structure(masking)

AVAILABLE = {
    'lars': LARSAlgorithm,
    'linear': LinearEquations,
    'nonlinear': NonlinearEquations,
}

DEFAULT = 'linear'
