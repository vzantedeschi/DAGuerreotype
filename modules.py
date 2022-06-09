from abc import ABC, abstractmethod

import torch
from torch import nn

from ranksp.ranksp import sparse_rank

from bernouilli import BernoulliSTERoot
from utils import get_optimizer

# ------------------------------------------------------------------------------------- ORDERINGS

class Masking(nn.Module, ABC):

    def __init__(self):

        super(Masking, self).__init__()

    @abstractmethod
    def forward(self):
        pass

    def l2(self):
        return 0.


class SparseMapMasking(Masking):

    def __init__(self, d, theta_init=None):

        super(SparseMapMasking, self).__init__()

        if theta_init is not None:
            theta = theta_init.clone().detach()

        else:
            theta = torch.zeros(d)

        self.theta = nn.Parameter(theta.unsqueeze(1), requires_grad=True)

        self.M = torch.triu(torch.ones((d, d)), diagonal=1)
    
    def forward(self, tmp=1e-5, init=False, max_iter=100):
        alphas, orderings = sparse_rank(self.theta / tmp, init=init, max_iter=max_iter)

        return alphas, self.M[orderings[..., None], orderings[:, None]]

    def l2(self):
        return torch.sum(self.theta ** 2)

# ----------------------------------------------------------------------------------- STRUCTURES

class Structure(nn.Module):

    def __init__(self, d, num_structures=1, initial_value=0.5):

        super(Structure, self).__init__()

        self.d = d
        self.num_structures = num_structures

        self.B = BernoulliSTERoot(
            (num_structures, d, d), 
            initial_value=initial_value * torch.ones((num_structures, d, d))
        ) # a Bernouilli variable per edge

    def forward(self, maskings):

        assert maskings.shape[0] == self.num_structures or self.num_structures == 1

        self.dag_mask = maskings

        sample_b = self.B() # sample 

        return maskings * sample_b

    def l0(self):

        masked_theta = self.B.theta * self.dag_mask

        return masked_theta.sum((-2, -1)) # one l0 per structure


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

        self.num_equations = num_equations

        self.W = nn.Parameter(
            torch.randn(num_equations, d, d),
            requires_grad=True
        )

    def forward(self, masked_x):

        assert masked_x.shape[0] == self.num_equations or self.num_equations == 1

        return torch.einsum("oncp,epc->onc", masked_x, self.W) # e = 0 or e = 1

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

    def __init__(self, d, num_structures, bernouilli_init=0.5):

        nn.Module.__init__(self)

        self.structure = Structure(d, num_structures, initial_value=bernouilli_init)
        self.equations = LinearEquations(d, num_structures)

    def forward(self, x, maskings):
        
        dags = self.structure(maskings)

        masked_x = torch.einsum("opc,np->oncp", dags, x) # for each ordering (o), data point (i) and node (c): vector v, with v_p = x_ip if p is potentially a parent of c, 0 otherwise

        x_hat = self.equations(masked_x)

        return x_hat

    def fit(self, x, maskings, loss, args):

        self.train()

        optimizer = get_optimizer(
            self.parameters(), name=args.optimizer, lr=args.lr
        )  # to update structure and equations

        # inner loop
        for inner_iters in range(args.num_inner_iters):
            
            x_hat = self(x, maskings) # (num_structures, n, d)

            objective = loss(x_hat, x, dim=(-2, -1))
            objective += args.pruning_reg * self.l0() + args.l2_reg * self.l2()

            objective.sum().backward()

            optimizer.step()
            optimizer.zero_grad()

            self.structure.B.project()

        self.eval()

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

    M = torch.triu(torch.ones((3, 3)), diagonal=1)
    Mp = M[inverse_ordering[..., None], inverse_ordering[:, None]]

    estimator = LinearL0Estimator(3, 1, bernouilli_init=1.)

    x_hat = estimator(x, Mp)

    assert (x_hat[0, :, 0] == 0.).all()

    ordering = torch.Tensor([1, 0, 2]).unsqueeze(0).long()
    inverse_ordering = torch.argsort(ordering)

    M = torch.triu(torch.ones((3, 3)), diagonal=1)
    Mp = M[inverse_ordering[..., None], inverse_ordering[:, None]]

    estimator = LinearL0Estimator(3, 1, bernouilli_init=1.)

    x_hat = estimator(x, Mp)

    assert (x_hat[0, :, 1] == 0.).all()


    x = torch.arange(1, 9).reshape(2, 4)

    ordering = torch.Tensor([[2, 0, 3, 1], [1, 3, 2, 0]]).long()
    inverse_ordering = torch.argsort(ordering)

    M = torch.triu(torch.ones((4, 4)), diagonal=1)
    Mp = M[inverse_ordering[..., None], inverse_ordering[:, None]]

    estimator = LinearL0Estimator(4, 2, bernouilli_init=1.)

    x_hat = estimator(x, Mp)

    assert (x_hat[0, :, 2] == 0.).all()
    assert (x_hat[1, :, 1] == 0.).all()