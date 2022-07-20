import warnings
from abc import ABC, abstractmethod

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC, LinearRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from torch import nn

from bernouilli import BernoulliSTERoot, BernoulliSTEOp
from ranksp.ranksp import sparse_rank
from utils import get_optimizer, get_variances


class _DagMod(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def initialize(cls, X, args, bilevel=False):
        raise NotImplementedError()


class Structure(_DagMod, ABC):

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
        alphas, orderings = sparse_rank(
            self.theta / self.smap_tmp,  # the actual parameter, this is a good place to do perturb & map insted
            init=self.smap_init, max_iter=self.smap_iter)
        # TODO possibly add eval branch that returns the MAP here, although - not so sure why we should take the MAP,
        #  can't we keep the probability distribution also at eval time?
        return alphas, self.M[orderings[..., None], orderings[:, None]], self.regularizer()


class TopKSparseMaxSVStructure(_ScoreVectorStructure):
    # TODO
    pass


# ----- sparsification methods

class _DagDownstreamMod(_DagMod, ABC):

    def bl_initialize(self, complete_dag): raise NotImplementedError()


class Sparsifier(_DagDownstreamMod, ABC):

    def forward(self, complete_dag):
        """

        Args:
            complete_dag: the tensor of complete dags (o x d x d binary tensor)

        Returns:
            Pair of (sparsified dags, regularizer)

        """
        raise NotImplementedError()


class NoSparsifier(Sparsifier):
    """Dummy class that does nothing :) """

    def forward(self, complete_dag): return complete_dag, 0

    def bl_initialize(self, complete_dag): return self

    @classmethod
    def initialize(cls, X, args, bilevel=False): return cls()


class _L0Sparsifier(Sparsifier, ABC):

    def __init__(self, l2_reg_strength, d, num_structures=None) -> None:
        super().__init__()
        self.l2_reg_strength = l2_reg_strength
        self.d = d
        self.num_structures = num_structures
        if self.num_structures:  # if this is none, it means we are in the bilevel setting
            self.pi = nn.Parameter(self._init_pi((self.num_structures, d, d)), requires_grad=True)

    @classmethod
    def initialize(cls, X, args, bilevel=False):
        d = X.shape[1]
        return cls(args.pruning_reg, d, None if bilevel else 1)

    def _init_pi(self, shape): raise NotImplementedError()

    def regularizer(self, complete_dag):
        masked_reg = complete_dag * torch.abs(self.pi)  # note, this is always a 3d tensor
        return self.l2_reg_strength * masked_reg.sum((1, 2))

    def bl_initialize(self, complete_dag):
        return self.__class__(self.l2_reg_strength, self.d, complete_dag.shape[0])


class BernoulliSTEL0Sparsifier(_L0Sparsifier):

    def __init__(self, l2_reg_strength, d, num_structures=None) -> None:
        super().__init__(l2_reg_strength, d, num_structures)
        if self.num_structures:
            self.op = BernoulliSTEOp((num_structures, d, d))

    def _init_pi(self, shape):
        return 0.5 * torch.ones(shape)

    def forward(self, complete_dags):
        self.pi.data.clamp(0.0, 1.0)  # make sure pi is in [0, 1] after updates (so no need of projecting!)
        # take the MAP when evaluating
        z = self.op(self.pi) if self.training else self.pi > 0.5
        # note that in joint optimization, z is still 1 sample! (d x d) matrix,
        # which will then be applied to all the complete dags!
        return complete_dags*z, self.regularizer(complete_dags)


class HardConcreteL0Sparsifier(_L0Sparsifier):
    # TODO
    pass


class L1Sparsifier(Sparsifier):
    # TODO (maybe....)
    pass


#  EQUATIONS ----------------

class Equations(_DagDownstreamMod, ABC):

    def forward(self, X, dags):
        """

        Args:
            X: the datapoints
            dags: tensor of dags (already "sparsified", possibly :) )

        Returns:
            The triplet [reconstructed X (one per dag), dag (possibly sparsified), regularizer)

        """
        raise NotImplementedError()

    def regularizer(self):
        return 0

    @staticmethod
    def is_complete_algorithm():
        return False

    @classmethod
    def initialize(cls, X, args, bilevel=False):
        if not bilevel:
            assert not cls.is_complete_algorithm(), f'{cls.__name__} only works in a bilevel setting'


class LARSAlgorithm(Equations):

    def __init__(self, d) -> None:
        super().__init__()
        self.d = d
        self.W = None

    def forward(self, X, dags):
        if self.training:
            self.fit(X, dags)
        return torch.einsum("np,opc->onc", X, self.W), self.W != 0, 0.

    def fit(self, X, dags):
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

    def bl_initialize(self, complete_dag):
        return self.__class__(self.d)  # not much to do here... maybe this is a bit redundant....

    @classmethod
    def initialize(cls, X, args, bilevel=False):
        super().initialize(X, args, bilevel)
        return cls(X.shape[1])

    @staticmethod
    def is_complete_algorithm():
        return True


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


class LinearEquations(Equations):

    def __init__(self, d, num_structures, l2_reg_strength=0.) -> None:
        super().__init__()
        self.l2_reg_strength = l2_reg_strength
        self.d = d
        self.num_structures = num_structures
        if self.num_structures:
            self.W = nn.Parameter(
                torch.zeros(self.num_structures, d, d), requires_grad=True
            )  # W[:, p, c] one weight from parent p to child c
            # W[0]'s column c reconstructs node c
        # TODO why no bias?

    def forward(self, X, dags):
        # reconstruct the child from the parents, one per dag!
        x1 = torch.einsum("oncp,opc->onc", masked_x(X, dags), self.W)
        return x1, dags, self.regularizer()

    def bl_initialize(self, complete_dag):
        return self.__class__(self.d, len(complete_dag), self.l2_reg_strength)

    @classmethod
    def initialize(cls, X, args, bilevel=False):
        super().initialize(X, args, bilevel)
        return cls(X.shape[1], None if bilevel else 1, args.l2_eq)

    def regularizer(self):
        return self.l2_reg_strength * torch.sum(self.W**2, dim=(-2, -1))


# ---- previous stuff  -----


# ------------------------------------------------------------------------------------- ORDERINGS


class Masking(nn.Module, ABC):
    def __init__(self):

        super(Masking, self).__init__()

    @abstractmethod
    def forward(self):
        pass

    def l2(self):
        return 0.0


class SparseMapMasking(Masking):
    def __init__(self, d, theta_init=None):

        super(SparseMapMasking, self).__init__()

        if theta_init is not None:
            m = theta_init.median()
            theta = (
                (theta_init - m).clone().detach()
            )  # to have both positive and negative values

        else:
            theta = torch.zeros(d)

        self.theta = nn.Parameter(theta.unsqueeze(1), requires_grad=True)

        self.M = torch.triu(torch.ones((d, d)), diagonal=1)

    def forward(self, tmp=1e-5, init=False, max_iter=100):
        alphas, orderings = sparse_rank(self.theta / tmp, init=init, max_iter=max_iter)

        return alphas, self.M[orderings[..., None], orderings[:, None]]

    def l2(self):
        return torch.sum(self.theta**2)


# ----------------------------------------------------------------------------------- STRUCTURES


class BernoulliStructure(nn.Module):
    def __init__(self, d, num_structures=1, initial_value=0.5):

        nn.Module.__init__(self)

        self.d = d
        self.num_structures = num_structures

        self.B = BernoulliSTERoot(
            (num_structures, d, d),
            initial_value=initial_value * torch.ones((num_structures, d, d)),
        )  # a Bernouilli variable per edge

    def forward(self, maskings):

        assert maskings.shape[0] == self.num_structures or self.num_structures == 1

        self.dag_mask = maskings

        sample_b = self.B()  # sample

        return maskings * sample_b

    def l0(self):

        masked_theta = self.B.theta * self.dag_mask

        return masked_theta.sum((-2, -1))  # one l0 per structure


# ------------------------------------------------------------------------------------ EQUATIONS


class EquationsOld(nn.Module, ABC):
    def __init__(self, d):

        super(EquationsOld, self).__init__()

        self.d = d

    @abstractmethod
    def forward(self, masked_x):
        pass

    @abstractmethod
    def l2(self):
        pass


class LinearEquationsOld(EquationsOld):
    def __init__(self, d, num_equations=1):

        super(LinearEquationsOld, self).__init__(d)

        self.num_equations = num_equations  # number of sets of structural equations,
        # for bilevel this is equal to the number of orderings, for joint this is equal to 1

        self.W = nn.Parameter(
            torch.randn(num_equations, d, d), requires_grad=True
        )  # W[:, p, c] one weight from parent p to child c
        # W[0]'s column c reconstructs node c

    def forward(self, masked_x):
        # [orderings, number of points, "child node", "parent node"],
        # [orderings, weight of parent node, weight of child node ]
        # result_onc = \sum_p X_oncp * W_opc
        return torch.einsum("oncp,opc->onc", masked_x, self.W)

    def l2(self):

        return torch.sum(self.W**2, dim=(-2, -1))  # one l2 per set of equations


class NonLinearEquationsOld(EquationsOld):
    def __init__(
        self, d, num_equations=1, hidden=2, activation=torch.nn.functional.leaky_relu
    ):

        super(NonLinearEquationsOld, self).__init__(d)

        self.num_equations = num_equations
        self.hidden = hidden

        self.W = nn.Parameter(
            torch.randn(num_equations, d, d, hidden)
            * 0.05,  # TODO: check for better value of std
            requires_grad=True,
        )

        self.bias = nn.Parameter(
            torch.zeros(num_equations, 1, d, hidden), requires_grad=True
        )

        self.activation = activation

        self.final_map = nn.Parameter(
            torch.randn(num_equations, d, hidden) * 0.05, requires_grad=True
        )

    def forward(self, masked_x):

        out = torch.einsum("oncp,opch->onch", masked_x, self.W)  #

        out = self.activation(out + self.bias)

        return torch.einsum("onch,och->onc", out, self.final_map)

    def l2(self):

        out = torch.sum(self.W**2, dim=(-3, -2, -1))  # one l2 per set of equations
        out += torch.sum(self.bias**2, dim=(-3, -2, -1))
        out += torch.sum(self.final_map**2, dim=(-2, -1))

        return out


# ---------------------------------------------------------------------------------- ESTIMATORS


class Estimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x, maskings, *args):
        pass

    @abstractmethod
    def get_structure(self, masking):
        pass

    def project(self):
        pass


class LARS(Estimator):
    def __init__(self, d, *args, **kwargs):

        self.d = d

    def __call__(self, x, *args):
        return torch.einsum("np,opc->onc", x, self.W)

    def fit(self, x, maskings, *args):

        LR = LinearRegression(normalize=False, n_jobs=1)
        LL = LassoLarsIC(criterion="bic", normalize=False)

        x_numpy = x.detach().numpy()
        masks_numpy = maskings.long().detach().numpy()

        self.W = np.zeros((len(masks_numpy), self.d, self.d))

        for m, mask in enumerate(masks_numpy):
            for target in range(self.d):

                covariates = np.nonzero(mask[:, target])[0]

                if len(covariates) > 0:  # if target is not a root node

                    LR.fit(x_numpy[:, covariates], x_numpy[:, target].ravel())
                    weight = np.abs(LR.coef_)

                    LL.fit(x_numpy[:, covariates] * weight, x_numpy[:, target].ravel())
                    self.W[m, covariates, target] = LL.coef_ * weight

            assert (self.W[m, mask == 0] == 0).all(), (self.W[m], mask)

        self.W = torch.from_numpy(self.W)

    def get_structure(self, *args):
        return self.W != 0


class NNL0Estimator(Estimator, nn.Module):
    def __init__(
        self,
        d,
        num_structures,
        bernouilli_init=0.5,
        linear=True,
        hidden=1,
        activation=torch.nn.functional.leaky_relu,
    ):

        nn.Module.__init__(self)
        # TODO: rename to sparsity
        self.structure = BernoulliStructure(
            d, num_structures, initial_value=bernouilli_init
        )

        if linear:
            self.equations = LinearEquationsOld(d, num_structures)
        else:
            self.equations = NonLinearEquationsOld(
                d, num_structures, hidden=hidden, activation=activation
            )

    def forward(self, x, maskings):

        dags = self.structure(maskings)

        masked_x = torch.einsum(
            "opc,np->oncp", dags, x
        )  # for each ordering (o), data point (i) and node (c): vector v, with v_p = x_ip if p is potentially a parent of c, 0 otherwise

        x_hat = self.equations(masked_x)

        return x_hat

    def fit(self, x, maskings, loss, args):

        self.train()

        optimizer = get_optimizer(
            self.parameters(), name=args.optimizer, lr=args.lr
        )  # to update structure and equations

        # inner loop
        for inner_iters in range(args.num_inner_iters):

            x_hat = self(x, maskings)  # (num_structures, n, d)

            objective = loss(x_hat, x, dim=(-2, -1))
            objective += args.pruning_reg * self.pruning() + args.l2_reg * self.l2()

            objective.sum().backward()

            optimizer.step()
            optimizer.zero_grad()

            self.project()

        self.eval()

    def project(self):
        self.structure.B.project()

    def pruning(self):
        return self.structure.l0()

    def l2(self):
        return self.equations.l2()

    def get_structure(self, masking):
        return self.structure(masking)


def get_estimator_cls(estimator: str):
    # TODO: add nonlinear estimator
    if estimator == "LARS":
        estimator_cls = LARS
    else:
        estimator_cls = NNL0Estimator
    return estimator_cls


if __name__ == "__main__":
    pass
