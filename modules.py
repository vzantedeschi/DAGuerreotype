from abc import ABC, abstractmethod

import torch
from torch import nn

from ranksp.ranksp import sparse_rank

from bernouilli import BernoulliSTERoot
from utils import get_optimizer

# ------------------------------------------------------------------------------------- ORDERINGS

class Ordering(nn.Module, ABC):

    def __init__(self):

        super(Ordering, self).__init__()

    @abstractmethod
    def forward(self):
        pass

    def l2(self):
        return 0.


class SparseMapOrdering(Ordering):

    def __init__(self, d, tmp=1e-5, init=False, theta_init=None, max_iter=100):

        super(SparseMapOrdering, self).__init__()

        self.tmp = tmp
        self.init = init
        self.max_iter = max_iter

        if theta_init is not None:
            theta = theta_init.clone().detach()

        else:
            theta = torch.zeros(d)

        self.theta = nn.Parameter(theta.unsqueeze(1), requires_grad=True)
    
    def forward(self):
        return sparse_rank(self.theta / self.tmp, init=self.init, max_iter=self.max_iter)

    def l2(self):
        return torch.sum(self.theta ** 2)

# ----------------------------------------------------------------------------------- STRUCTURES

class Structure(nn.Module):

    def __init__(self, d, num_structures=1, initial_value=0.5):

        super(Structure, self).__init__()

        self.d = d
        self.num_structures = num_structures

        self.M = nn.Parameter(
            torch.triu(torch.ones((d, d)), diagonal=1), 
            requires_grad=False
        )  # sets diagonal and lower triangle to 0 

        self.B = BernoulliSTERoot(
            (num_structures, d, d), 
            initial_value=initial_value * torch.ones((num_structures, d, d))
        ) # a Bernouilli variable per edge

    def forward(self, orderings):

        assert orderings.shape == (self.num_structures, self.d), (orderings.shape, (self.num_structures, self.d))

        self.dag_mask = self.M[orderings[..., None], orderings[:, None]] # ensure maximal graph is a DAG

        sample_b = self.B() # sample 

        return self.dag_mask * sample_b

    def l0(self):

        masked_theta = self.B.theta * self.dag_mask

        return masked_theta.sum((-2, -1)) # one l0 per structure


# ------------------------------------------------------------------------------------ EQUATIONS

class Equations(nn.Module, ABC):

    def __init__(self):

        super(Equations, self).__init__()

    @abstractmethod
    def forward(self, masked_x):
        pass

    @abstractmethod
    def l2(self):
        pass

class LinearEquations(Equations):

    def __init__(self, d, num_equations=1):

        super(Equations, self).__init__()

        self.d = d
        self.num_equations = num_equations

        self.W = nn.Parameter(
            torch.randn(num_equations, d, d),
            requires_grad=True
        )

    def forward(self, masked_x):

        assert masked_x.shape[0] == self.num_equations or self.num_equations == 1

        return torch.einsum("oncp,epc->onc", masked_x, self.W) # e = 1 or e = o

    def l2(self):

        return torch.sum(self.W ** 2, dim=(-2, -1)) # one l2 per structure

# ---------------------------------------------------------------------------------- ESTIMATORS

class Estimator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x, orderings):
        pass

class LinearL0Estimator(Estimator, nn.Module):

    def __init__(self, d, num_structures, num_equations=1, bernouilli_init=0.5):

        nn.Module.__init__(self)

        self.structure = Structure(d, num_structures, initial_value=bernouilli_init)
        self.equations = LinearEquations(d, num_equations)

    def forward(self, x, orderings):
        
        mask = self.structure(orderings) # masks to remove dependencies (i) inconstistent with the orderings or (ii) pruned out

        masked_x = torch.einsum("opc,np->oncp", mask, x) # for each ordering (o), data point (i) and node (c): vector v, with v_p = x_ip if p is potentially a parent of c, 0 otherwise

        x_hat = self.equations(masked_x)

        return x_hat

    def fit(self, x, orderings, loss, args):

        self.train()

        optimizer = get_optimizer(
            self.parameters(), name=args.optimizer, lr=args.lr
        )  # to update structure and equations

        # inner loop
        for inner_iters in range(args.num_inner_iters):
            
            x_hat = self(x, orderings) # (num_structures, n, d)

            objective = loss(x_hat, x, dim=(-2, -1))
            objective += args.pruning_reg * self.l0() + args.l2_reg * self.l2()

            objective.sum().backward()

            optimizer.step()
            optimizer.zero_grad()

            self.structure.B.project()

    def l0(self):
        return self.structure.l0()

    def l2(self):
        return self.equations.l2()

if __name__ == "__main__":

    from utils import init_seeds

    init_seeds(39)
    x = torch.arange(1, 7).reshape(2, 3)

    ordering = torch.Tensor([0, 1, 2]).unsqueeze(0).long()
    inverse_ordering = torch.argsort(ordering)

    estimator = LinearL0Estimator(3, 1, 1, bernouilli_init=1.)

    x_hat = estimator(x, inverse_ordering)

    assert (x_hat[0, :, 0] == 0.).all()

    ordering = torch.Tensor([1, 0, 2]).unsqueeze(0).long()
    inverse_ordering = torch.argsort(ordering)

    estimator = LinearL0Estimator(3, 1, 1, bernouilli_init=1.)

    x_hat = estimator(x, inverse_ordering)

    assert (x_hat[0, :, 1] == 0.).all()


    x = torch.arange(1, 9).reshape(2, 4)

    ordering = torch.Tensor([2, 0, 3, 1]).unsqueeze(0).long()
    inverse_ordering = torch.argsort(ordering)

    estimator = LinearL0Estimator(4, 1, 1, bernouilli_init=1.)

    x_hat = estimator(x, inverse_ordering)

    assert (x_hat[0, :, 2] == 0.).all()