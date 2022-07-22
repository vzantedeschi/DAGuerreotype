import typing
from abc import ABC

import torch


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
        pass

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
        uni = torch.rand(self.shape)  # , generator=generator)
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
        self.pi = 0.5 * torch.ones((num_structures, self.d, self.d))

    def forward(self, complete_dags):
        self.pi.data.clamp(0.0, 1.0)  # make sure pi is in [0, 1] after updates (so no need of projecting!)
        # take the MAP when evaluating
        z = self.op(self.pi)
        # note that in joint optimization, z is still 1 sample! (d x d) matrix,
        # which will then be applied to all the complete dags!
        return complete_dags*z, self.regularizer(complete_dags)


class HardConcreteL0Sparsifier(_L0Sparsifier):
    # TODO
    pass


class L1Sparsifier(Sparsifier):
    # TODO (maybe....)
    pass


AVAILABLE = {
    'l0_ber_ste': BernoulliSTEL0Sparsifier,
    'l0_hc': HardConcreteL0Sparsifier,
    'none': NoSparsifier
}

DEFAULT = 'l0_ber_ste'
