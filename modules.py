from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from bernouilli import BernoulliSTERoot

# ------------------------------------------------------------------------------------- ORDERINGS

class Ordering(nn.Module, ABCMeta):

    def __init__(self):

        super(Ordering, self).__init__()

    @abstractmethod
    def forward(self):
    	pass


class SparseMapOrdering(Ordering):

    def __init__(self, tmp=1e-5, init=False, theta_init=None, max_iter=100):

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
    	return sparse_rank(self.theta.cpu() / self.tmp, init=self.init, max_iter=self.max_iter)

# ----------------------------------------------------------------------------------- STRUCTURES

class Structure(nn.Module):

    def __init__(self, d, num_structures=1):

        super(Structure, self).__init__()

        self.d = d
        self.num_structures = num_structures

        self.M = nn.Parameter(
        	torch.triu(torch.ones((d, d)), diagonal=1), 
        	requires_grad=False
        )  # sets diagonal and lower triangle to 0 

        self.B = BernoulliSTERoot(
        	(num_structures, d, d), 
        	initial_value=0.5 * torch.ones((d, d))
        ) # a Bernouilli variable per edge

    def forward(self, orderings):

    	assert orderings.shape == (self.num_structures, self.d, self.d)

    	self.dag_mask = self.M[orderings[..., None], orderings[:, None]] # ensure maximal graph is a DAG

    	sample_b = self.B() # sample 

    	return self.dag_mask * sample_b

    def l0(self):

        masked_theta = self.B.theta * self.dag_mask

        return masked_theta.sum((-2, -1))


# ------------------------------------------------------------------------------------ EQUATIONS

class Equations(nn.Module, ABCMeta):

    def __init__(self):

        super(Equations, self).__init__()

    @abstractmethod
    def forward(self, masked_x):
    	pass

class LinearEquations(Equations):

    def __init__(self, d, num_equations=1):

        super(Equations, self).__init__()

        self.d = d

        self.W = nn.Parameter(
            torch.Tensor(num_equations, d, d),
            requires_grad=True
        )

    def forward(self, masked_x):

    	return torch.einsum("cnd,cdd->cnd", masked_x, self.W)
