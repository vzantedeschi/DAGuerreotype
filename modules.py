import warnings
from abc import ABC, abstractmethod

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC, LinearRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from torch import nn

from bernouilli import BernoulliSTERoot
from ranksp.ranksp import sparse_rank
from utils import get_optimizer, get_variances


class _DagMod(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def initialize(cls, X, args, bilevel):
        pass

    def regularizer(self): return 0


class Structure(_DagMod):

    def forward(self):
        """This is for the unconditional computation of a probability distribution over complete DAG structures."""
        pass


class _ScoreVectorStructure(Structure):

    def __init__(self, d, theta_init) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(theta_init.unsqueeze(1), requires_grad=True)
        self.M = torch.triu(torch.ones((d, d)), diagonal=1)

    @classmethod
    def initialize(cls, X, args, bilevel):
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
        return cls(d, theta_init)


class SparseMapSVStructure(_ScoreVectorStructure):
    """
    Class for learning the score vector with the sparseMAP operator.
    """

    def forward(self):
        # call the sparseMAP rank procedure, it returns a vector of probabilities and one of sorted indices,
        # FIXME re-check if indieces are in ascending or descending order!!!!!!
        alphas, orderings = sparse_rank(
            self.theta / self.args.smap_tmp,  # the actual parameter, this is a good place to do perturb & map insted
            init=self.args.smap_init, max_iter=self.args.smap_iter)
        # TODO possibly add eval branch that returns the MAP here, although - not so sure why we should take the MAP,
        #  can't we keep the probability distribution also at eval time?
        return alphas, self.M[orderings[..., None], orderings[:, None]]

    def regularizer(self):
        return self.args.l2_reg * (self.theta**2).sum()


class TopKSparseMaxSVStructure(_ScoreVectorStructure):
    # TODO
    pass


class Sparsifier(_DagMod):
    pass

class _L0Sparsifier(Sparsifier):

    def __init__(self) -> None:
        super().__init__()
        self.pi = None
        self.complete_dags = None

    def regularizer(self):
        """This is the """
        masked_reg = torch.abs(self.pi) * self.complete_dags
        return self.args.pruning_reg * masked_reg.sum((1, 2))


class BernoulliSTEL0Sparsifier(_L0Sparsifier):

    def __init__(self) -> None:
        super().__init__()
        self.ber_ste_op = None

    def initialize(self, X, args, bilevel):
        super().initialize(X, args, False)


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

class Equations(_DagMod):
    pass

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


class Equations(nn.Module, ABC):
    def __init__(self, d):

        super(Equations, self).__init__()

        self.d = d

    @abstractmethod
    def forward(self, masked_x):
        pass

    @abstractmethod
    def l2(self):
        pass


class LinearEquations(Equations):
    def __init__(self, d, num_equations=1):

        super(LinearEquations, self).__init__(d)

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


class NonLinearEquations(Equations):
    def __init__(
        self, d, num_equations=1, hidden=2, activation=torch.nn.functional.leaky_relu
    ):

        super(NonLinearEquations, self).__init__(d)

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
            self.equations = LinearEquations(d, num_structures)
        else:
            self.equations = NonLinearEquations(
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
