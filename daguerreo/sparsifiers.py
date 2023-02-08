import math
import typing
from abc import ABC

import torch
from torch.autograd import Variable
from torch.nn import functional as F


class Sparsifier(torch.nn.Module):

    @classmethod
    def initialize(cls, X, args, joint=False): raise NotImplementedError()

    def forward(self, complete_dag):
        """

        Args:
            complete_dag: the tensor of complete dags (o x d x d binary tensor)

        Returns:
            Pair of (sparsified dags, regularizer)

        """
        raise NotImplementedError()

    def init_parameters(self, num_structures): raise NotImplementedError()


class NoSparsifier(Sparsifier):
    """Dummy class that does nothing :) """

    def init_parameters(self, num_structures):
        return self

    def forward(self, complete_dag): return complete_dag, 0

    @classmethod
    def initialize(cls, X, args, joint=False): return cls()


class _L0Sparsifier(Sparsifier, ABC):

    def __init__(self, l2_reg_strength, d, num_structures=None) -> None:
        super().__init__()
        self.l2_reg_strength = l2_reg_strength
        self.d = d
        self.num_structures = num_structures
        self.pi = None
        if self.num_structures: self.init_parameters(self.num_structures)

    def init_parameters(self, num_structures): raise NotImplementedError()

    @classmethod
    def initialize(cls, X, args, joint=False):
        d = X.shape[1]
        return cls(args.pruning_reg, d, 1 if joint else None)

    def regularizer(self, complete_dag):
        masked_reg = complete_dag * torch.abs(self.pi)  # note, this is always a 3d tensor
        return self.l2_reg_strength * masked_reg.sum((1, 2))


class BernoulliRV:
    def __init__(self, shape):
        self.shape = shape

    def _sample(self, theta):
        """
        Draws one sample of the rv with success parameter theta

        :param theta: the success parameter in [0, 1]
        :return: a sample
        """
        uni = torch.rand(self.shape, device=theta.device)  # , generator=generator)
        return (torch.sign(theta - uni) + 1) / 2

    def sampler(self):
        """
        :return: a sampler for the rv (a callable)
        """
        return lambda theta: self._sample(theta)

    @property
    def ndim(self):
        """
        :return: dimensionality of this rv
        """
        return len(self.shape)


# ------------------------------------------------------------------------------------------
# ste function: takes in sampler and returns a lambda function(which is a callable)
# the lambda function takes in sampler and theta (parameter for bernoulli) and pass those
# to _StraightThroughEstimator.apply(). The returned callable implements forward and backward function


def ste(sampler: typing.Callable):
    """
    Function to call for initializing a straight-through estimator

    :param sampler: a function that returns a sample of the random variable
    :return: a callable that implements the forward and the backward pass
    """
    return lambda theta: _StraightThroughEstimator.apply(theta, sampler)


# It is a subclass of a torch.autograd.Function. We can extend autograd with this
# subclassing to implement our own custom autograd.
# forward function does forward pass i.e. just samples bern rv
# backward function does backprop. The gradient of the function
# For STE, we want grad_output


class _StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    # computes output tensors from input tensors
    # forward returns 0, 1 suing theta as parameter for Bern
    def forward(ctx, theta, sampler):
        return sampler(theta)

    @staticmethod
    # we get tensor containing the gradient of loss
    # w.r.t the output (grad_output)and want to caculate gradient for loss
    # w.r.t input. STE approximates gradient of loss w.r.t. the input as
    # gradient of loss w.r.t. the output
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class BernoulliSTEOp(torch.nn.Module):
    def __init__(self, shape):
        """

        Args:
            shape:
        """
        super().__init__()
        self.dist = BernoulliRV(shape)
        self.ste_layer = ste(self.dist.sampler())

    def forward(self, theta):
        if theta.ndim == self.dist.ndim:
            return self.ste_layer(theta) if self.training else theta > 0.5
        else:
            return torch.stack([self.forward(t) for t in theta])  # parallelization?


class BernoulliSTEL0Sparsifier(_L0Sparsifier):

    def __init__(self, l2_reg_strength, d, num_structures=None) -> None:
        self.op = None
        super().__init__(l2_reg_strength, d, num_structures)

    def init_parameters(self, num_structures):
        self.op = BernoulliSTEOp((num_structures, self.d, self.d))
        self.pi = torch.nn.Parameter(
            0.5 * torch.ones((num_structures, self.d, self.d))
        )
        return self

    def forward(self, complete_dags):
        self.pi.data.clamp(0.0, 1.0)  # make sure pi is in [0, 1] after updates (so no need of projecting!)

        z = self.op(self.pi)  # this takes the MAP when eval

        # note that in joint optimization, z is still 1 sample! (d x d) matrix,
        # which will then be applied to all the complete dags!
        return complete_dags*z, self.regularizer(complete_dags)


class HardConcreteRV:

    def __init__(self, shape, epsilon=1e-6, limit_a=-0.1, limit_b=1.1, temperature=2./3.):
        self.temperature = temperature
        self.shape = shape
        self.epsilon = epsilon
        self.limit_a = limit_a
        self.limit_b = limit_b

    def get_eps(self, size, device=None):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.nn.init.uniform_(torch.empty(size, device=device), self.epsilon, 1 - self.epsilon)
        eps = Variable(eps)
        return eps

    # in the functions below, theta is the log alpha in the paper

    def cdf_qz(self, x, theta):
        """Implements the CDF of the 'stretched' concrete distribution"""
        # didn't fully understan this... see bewlow
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)

        # from the paper, this computed in 0 should be  [a = self.limit_a, b = self.limit_b]
        # sigmoid(theta - temperature * (log( - a) - log(b) )
        # see eq 12
        #
        # if x = 0 then xn = - a / (b - a)
        # logits = log( -a ) - log( b-a) - log ( b) + log(b -a) = log(-a) - log(b)
        # so... it looks like there's the opposite sing in the expression below..
        # shouldn't it be  theta - logists * temp ?? (according to the paper?)

        # aanyway... it seems numerically correct, in the sense that it decreases
        # as theta -> - infinity
        return torch.sigmoid(logits * self.temperature - theta)  #.clamp(min=self.epsilon, max=1 - self.epsilon)

    def quantile_concrete(self, theta, eps):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + theta) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def sample(self, theta):
        """
        Draws one sample of the rv with parameter theta

        :param theta: the success parameter
        :return: a sample
        """
        eps = self.get_eps(self.shape, device=theta.device)
        z = self.quantile_concrete(theta, eps)
        return F.hardtanh(z, min_val=0, max_val=1)

    def map(self, theta):
        pi = torch.sigmoid(theta)
        return F.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    @property
    def ndim(self):
        """
        :return: dimensionality of this rv
        """
        return len(self.shape)


class HardConcreteL0Sparsifier(_L0Sparsifier):

    def __init__(self, l2_reg_strength, d, num_structures=None) -> None:
        self.rv = None
        super().__init__(l2_reg_strength, d, num_structures)

    def init_parameters(self, num_structures):
        shape = (num_structures, self.d, self.d)
        self.pi = torch.nn.Parameter(torch.empty(shape))
        self.pi.data.normal_(0., 1e-2)
        self.rv = HardConcreteRV(shape)

        return self

    def forward(self, complete_dags):
        # self.pi.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

        op = self.rv.sample if self.training else self.rv.map
        z = op(self.pi)

        return complete_dags*z, self.regularizer(complete_dags)

    def regularizer(self, complete_dag):
        # logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        return self.l2_reg_strength * torch.sum((1 - self.rv.cdf_qz(0, self.pi)) * complete_dag)
        # return super().regularizer(complete_dag)


class L1Sparsifier(Sparsifier):
    # TODO (maybe....)
    pass


AVAILABLE = {
    'l0_ber_ste': BernoulliSTEL0Sparsifier,
    'l0_hc': HardConcreteL0Sparsifier,
    'none': NoSparsifier
}

DEFAULT = 'l0_ber_ste'
