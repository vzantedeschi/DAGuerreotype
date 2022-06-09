import wandb

import torch
from torch import nn
from tqdm import tqdm

from modules import SparseMapOrdering, LinearL0Estimator
from utils import get_optimizer

class Daguerreo():

    def __init__(self, d, estimator_cls=LinearL0Estimator):

        # TODO: allow different init for theta
        self.smap_ordering = SparseMapOrdering(d, theta_init=None)

        self.estimator_cls = estimator_cls
        self.d = d

    def bilevel_optimization(self, x, loss, args):

        log_dict = {}

        smap_optim = get_optimizer(self.smap_ordering.parameters(), name=args.optimizer, lr=args.lr_theta)  # to update sparsemap parameters

        pbar = tqdm(range(args.num_epochs))

        # outer loop
        for epoch in pbar:

            smap_optim.zero_grad()

            alphas, orderings = self.smap_ordering(args.smap_tmp, args.smap_init, args.smap_iter)
            num_orderings = len(alphas)

            # inner problem: learn regressor
            self.estimator = self.estimator_cls(self.d, num_orderings, num_orderings)
            self.estimator.fit(x, orderings, loss, args)

            x_hat = self.estimator(x, orderings)
            fidelity_objective = loss(x_hat, x, dim=(-2, -1))

            # this is the outer loss (i.e. the loss that depends only on the ranking params)
            rec_loss = alphas @ fidelity_objective.detach()
            theta_norm = self.smap_ordering.l2()

            outer_objective = rec_loss + args.l2_reg * theta_norm

            outer_objective.backward()

            smap_optim.step()

            pbar.set_description(f"outer objective {outer_objective.item():.2f} | np {num_orderings} | theta norm {theta_norm:.4f}")

            log_dict = {
                "epoch": epoch,
                "number of orderings": num_orderings,
                "outer objective": outer_objective.item(),
                "expected loss": rec_loss.item(),
            }
            wandb.log(log_dict)

        return log_dict

    def joint_optimization(self, x, loss, args):
        pass

    def fit_mode(self, x, loss, args):

        alphas, self.mode_ordering = self.smap_ordering() # with default values, returns MAP
        assert len(alphas) == 1
        
        self.mode_estimator = self.estimator_cls(self.d, 1, 1) # one structure, one set of equations
        self.mode_estimator.fit(x, self.mode_ordering, loss, args)

    def get_binary_adj(self):

        return self.mode_estimator.structure(self.mode_ordering).squeeze(0) 