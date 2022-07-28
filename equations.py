import math
import warnings
from abc import ABC
from itertools import chain

import torch.nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LassoLarsIC
import numpy as np
from torch import nn

import utils
from sparsifiers import Sparsifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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

    @classmethod   # TODO this should be the other way around... namely construct the parser and args from hypers!
    def _hps_from_args(cls, args):
        # todo also here we'd better decouple the hyperparameters
        return {
            'optimizer': lambda _vrs: utils.get_optimizer(_vrs, name=args.eq_optimizer, lr=args.lr),
            'n_iters': args.num_inner_iters,
            'l2_reg_strength': args.l2_eq
        }

    def fit(self, X, complete_dags, dag_sparsifier, loss):
        super(ParametricGDFitting, self).fit(X, complete_dags, dag_sparsifier, loss)
        n_dags = len(complete_dags)
        dag_sparsifier.init_parameters(n_dags)
        self.init_parameters(n_dags)

        inner_opt = self.optimizer(chain(self.parameters(), dag_sparsifier.parameters()))

        for inner_iters in range(self.n_iters):
            inner_opt.zero_grad()

            dags, sparsifier_reg = dag_sparsifier(complete_dags)
            x_hat, dags, equations_reg = self(X, dags)

            inner_objective = loss(x_hat, X, dim=(1, 2)) + sparsifier_reg + equations_reg

            inner_objective.sum().backward()
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

    def forward(self, X, dags):
        mx = masked_x(X, dags)
        out = self._forward_impl(mx)
        return out, dags, self.regularizer()

    def _forward_impl(self, masked_X):
        raise NotImplementedError()


def _initialize_param(*shape, initializer=torch.zeros):
    return nn.Parameter(initializer(shape), requires_grad=True)


class LinearEquations(ParametricGDFitting):

    def __init__(self, d, num_structures, l2_reg_strength, optimizer, n_iters) -> None:
        self.W = None
        super().__init__(d, num_structures, l2_reg_strength, optimizer, n_iters)

    def init_parameters(self, num_structures):
        # W[:, p, c] one weight from parent p to child c
        # W[0]'s column c reconstructs node c
        # TODO why no bias?
        self.W = _initialize_param(num_structures, self.d, self.d)

    def _forward_impl(self, masked_X):
        # reconstruct the child from the parents, one per dag!
        return torch.einsum("oncp,opc->onc", masked_X, self.W)


class NonlinearEquations(ParametricGDFitting):
    """
    Implementation of a one-hidden-layer feed-forward neural net that preserves a given dag structure.
    """

    def __init__(self, d, num_structures, l2_reg_strength, optimizer, n_iters,
                 hidden_units, activation=torch.nn.functional.leaky_relu) -> None:
        self.hidden_units = hidden_units
        self.activation = activation
        self.W1 = None
        self.b1 = None
        self.W2 = None
        # todo why no b2?
        super().__init__(d, num_structures, l2_reg_strength, optimizer, n_iters)

    @classmethod
    def _hps_from_args(cls, args):
        return super()._hps_from_args(args) | {
            'hidden_units': args.hidden,
            # TODO activation seems missing in args :)
        }

    def init_parameters(self, num_structures):
        self.W1 = _initialize_param(num_structures, self.d, self.d, self.hidden_units)
        torch.nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))  # taken from torch.nn.Linear

        self.b1 = _initialize_param(num_structures, 1, self.d, self.hidden_units)
        self.W2 = _initialize_param(num_structures, self.d, self.hidden_units)

    def _forward_impl(self, masked_X):
        # computes hidden layer, one per dag, per example, child node, hidden unit (4-d tensor)
        out = torch.einsum("oncp,opch->onch", masked_X, self.W1)
        out = self.activation(out + self.b1)

        # computes output (summing over the hidden dimension)
        out = torch.einsum("onch,och->onc", out, self.W2)
        return out


AVAILABLE = {
    'lars': LARSAlgorithm,
    'linear': LinearEquations,
    'nonlinear': NonlinearEquations,
}

DEFAULT = 'linear'
