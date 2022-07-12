import torch
import wandb
from torch import nn
from tqdm import tqdm

from modules import NNL0Estimator, SparseMapMasking
from utils import get_optimizer, get_variances


class Daguerreo(nn.Module):
    def __init__(
        self, d, X_torch, smap_init_theta:str, estimator_cls=NNL0Estimator, estimator_kwargs=None
    ):

        super(Daguerreo, self).__init__()

        if smap_init_theta == "variances":
            smap_init = get_variances(X_torch)
        else:
            smap_init = None

        self.smap_masking = SparseMapMasking(d, theta_init=smap_init)

        self.estimator_cls = estimator_cls

        if estimator_kwargs is None:
            self.estimator_kwargs = {}
        else:
            assert type(estimator_kwargs) is dict
            self.estimator_kwargs = estimator_kwargs

        self.d = d

    def bilevel_optimization(self, x, loss, args):

        log_dict = {}

        smap_optim = get_optimizer(
            self.smap_masking.parameters(), name=args.optimizer, lr=args.lr_theta
        )  # to update sparsemap parameters

        pbar = tqdm(range(args.num_epochs))

        # outer loop
        for epoch in pbar:

            smap_optim.zero_grad()

            alphas, maskings = self.smap_masking(
                args.smap_tmp, args.smap_init, args.smap_iter
            )
            num_orderings = len(alphas)

            # inner problem: learn regressor
            self.estimator = self.estimator_cls(
                self.d, num_orderings, **self.estimator_kwargs
            )
            self.estimator.fit(x, maskings, loss, args)

            x_hat = self.estimator(x, maskings)
            fidelity_objective = loss(x_hat, x, dim=(-2, -1))

            # this is the outer loss (i.e. the loss that depends only on the ranking params)
            rec_loss = alphas @ fidelity_objective.detach()
            theta_norm = self.smap_masking.l2()

            outer_objective = rec_loss + args.l2_reg * theta_norm

            outer_objective.backward()

            smap_optim.step()

            pbar.set_description(
                f"outer objective {outer_objective.item():.2f} | np {num_orderings} | theta norm {theta_norm:.4f}"
            )

            log_dict = {
                "epoch": epoch,
                "number of orderings": num_orderings,
                "outer objective": outer_objective.item(),
                "expected loss": rec_loss.item(),
            }
            wandb.log(log_dict)

        return log_dict

    def joint_optimization(self, x, loss, args):

        log_dict = {}

        self.estimator = self.estimator_cls(
            self.d, 1, **self.estimator_kwargs
        )  # init single structure

        optimizer = get_optimizer(self.parameters(), name=args.optimizer, lr=args.lr)

        pbar = tqdm(range(args.num_epochs))

        # outer loop
        for epoch in pbar:

            optimizer.zero_grad()

            alphas, maskings = self.smap_masking(
                args.smap_tmp, args.smap_init, args.smap_iter
            )
            num_orderings = len(alphas)

            x_hat = self.estimator(x, maskings)
            self.estimator.project()

            exp_loss = alphas @ loss(x_hat, x, dim=(-2, -1))
            exp_l0 = alphas @ self.estimator.pruning()
            theta_norm = self.smap_masking.l2()
            l2 = theta_norm + self.estimator.l2().sum()

            objective = exp_loss + args.pruning_reg * exp_l0 + args.l2_reg * l2

            objective.backward()

            optimizer.step()

            pbar.set_description(
                f"objective {objective.item():.2f} | np {num_orderings} | theta norm {theta_norm:.4f}"
            )

            log_dict = {
                "epoch": epoch,
                "number of orderings": num_orderings,
                "objective": objective.item(),
                "expected loss": exp_loss.item(),
                "expected l0": exp_l0.item(),
            }
            wandb.log(log_dict)

        return log_dict

    def fit_mode(self, x, loss, args):

        # todo: use sort instead
        (
            alphas,
            self.mode_masking,
        ) = self.smap_masking()  # with default values, returns MAP
        assert len(alphas) == 1

        self.mode_estimator = self.estimator_cls(
            self.d, 1, **self.estimator_kwargs
        )  # one structure
        self.mode_estimator.fit(x, self.mode_masking, loss, args)

    def get_binary_adj(self):

        return self.mode_estimator.get_structure(self.mode_masking).squeeze(0)
