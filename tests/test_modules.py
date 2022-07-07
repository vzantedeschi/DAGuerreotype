import unittest

import torch

from modules import NNL0Estimator, NonLinearEquations


class TestNNL0Estimator(unittest.TestCase):
    def test_case_one(self):
        x = torch.arange(1, 7).reshape(2, 3)

        ordering = torch.Tensor([0, 1, 2]).unsqueeze(0).long()
        inverse_ordering = torch.argsort(ordering)

        M = torch.triu(torch.ones((3, 3)), diagonal=1)
        Mp = M[inverse_ordering[..., None], inverse_ordering[:, None]]

        estimator = NNL0Estimator(3, 1, bernouilli_init=1.0)

        x_hat = estimator(x, Mp)

        assert (x_hat[0, :, 0] == 0.0).all()

    def test_case_two(self):
        x = torch.arange(1, 7).reshape(2, 3)

        ordering = torch.Tensor([1, 0, 2]).unsqueeze(0).long()
        inverse_ordering = torch.argsort(ordering)

        M = torch.triu(torch.ones((3, 3)), diagonal=1)
        Mp = M[inverse_ordering[..., None], inverse_ordering[:, None]]

        estimator = NNL0Estimator(3, 1, bernouilli_init=1.0)

        x_hat = estimator(x, Mp)

        assert (x_hat[0, :, 1] == 0.0).all()

    def test_case_three(self):
        x = torch.arange(1, 9).reshape(2, 4)

        ordering = torch.Tensor([[2, 0, 3, 1], [1, 3, 2, 0]]).long()
        inverse_ordering = torch.argsort(ordering)

        M = torch.triu(torch.ones((4, 4)), diagonal=1)
        Mp = M[inverse_ordering[..., None], inverse_ordering[:, None]]

        estimator = NNL0Estimator(4, 2, bernouilli_init=1.0)

        x_hat = estimator(x, Mp)

        assert (x_hat[0, :, 2] == 0.0).all()
        assert (x_hat[1, :, 1] == 0.0).all()

    def test_case_four(self):
        x = torch.arange(1, 9).reshape(2, 4)

        ordering = torch.Tensor([[2, 0, 3, 1], [1, 3, 2, 0]]).long()
        inverse_ordering = torch.argsort(ordering)

        M = torch.triu(torch.ones((4, 4)), diagonal=1)
        Mp = M[inverse_ordering[..., None], inverse_ordering[:, None]]

        nonlinear_equations = NonLinearEquations(4, num_equations=1, hidden=10)

        masked_x = torch.einsum(
            "opc,np->oncp", Mp, x
        )  # for each ordering (o), data point (i) and node (c): vector v, with v_p = x_ip if p is potentially a parent of c, 0 otherwise

        x_hat = nonlinear_equations(masked_x)

        assert len(torch.unique(x_hat[0, :, 2])) == 1
        assert len(torch.unique(x_hat[1, :, 1])) == 1
