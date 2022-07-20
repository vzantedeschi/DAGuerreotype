import torch
import wandb
from torch import nn
from tqdm import tqdm

from typing import Type, Optional

from modules import NNL0Estimator, SparseMapMasking, Structure, Sparsifier, \
    Equations
from utils import get_optimizer, get_variances
from itertools import chain


# TODO do we need this to subclass nn.Module?
class Daguerro(torch.nn.Module):

    def __init__(self, structure: Structure,
                 sparsifier: Sparsifier,
                 equations: Equations) -> None:
        super(Daguerro, self).__init__()
        self.structure = structure
        self.sparsifier = sparsifier
        self.equations = equations

    @staticmethod
    def initialize(X, args, bilevel=False):
        dag_cls = DaguerroBilevel if bilevel else DaguerroJoint
        # INFO: initialization is deferred to the specific methods to deal with all the modules'
        # hyperparameters inplace
        structure = args.structure.initialize(X, args)  # there's no difference between bilevel and joint opt here
        sparsifier = args.sparsifier.initialize(X, args, bilevel)
        equations = args.equations.initialize(X, args, bilevel)
        return dag_cls(structure, sparsifier, equations)

    def forward(self, X, loss, args):
        # TODO remove args from here and push everything to the initialize method,
        #  the rationale is that in this way one can also run the algorithm not form the CLI!
        #  as it stands it is very complicated to do that!
        if self.training:
            return self._learn(X, loss, args)
        else:
            return self._eval(X, loss, args)

    def _learn(self, X, loss, args): raise NotImplementedError()

    def _eval(self, X, loss, args): raise NotImplementedError()


class DaguerroJoint(Daguerro):

    def _learn(self, X, loss, args):
        # here all should be initialized
        log_dict = {}

        optimizer = get_optimizer(self.parameters(), name=args.optimizer, lr=args.lr)

        pbar = tqdm(range(args.num_epochs))

        # outer loop
        for epoch in pbar:

            optimizer.zero_grad()

            # INFO: main calls, note that here all the regularization terms are computed by the respective modules
            alphas, complete_dags, structure_reg = self.structure()
            dags, sparsifier_reg = self.sparsifier(complete_dags)
            x_hat, dags, equations_reg = self.equations(X, dags)

            rec_loss = loss(x_hat, X, dim=(-2, -1))

            #  here we weight also the regularizers by the alphas, can discuss about this...
            #  but I think this is correct. The structure regularizer is instead unweighted as it is a global one
            objective = alphas @ (rec_loss + sparsifier_reg + equations_reg) + structure_reg

            objective.backward()

            optimizer.step()

            pbar.set_description(
                f"objective {objective.item():.2f} | ns {len(complete_dags)} | "
                f"theta norm {self.structure.theta.norm():.4f}"
            )

            log_dict = {
                "epoch": epoch,
                # "number of orderings": num_orderings,
                "objective": objective.item(),
                # "expected loss": exp_loss.item(),
                # "expected l0": exp_l0.item(),
            }
            # TODO should be able to exclude wandb if don't want to use it :)
            wandb.log(log_dict)

            print(self.structure.theta)

        return log_dict

    def _eval(self, X, loss, args):
        pass


class DaguerroBilevel(Daguerro):

    def _learn(self, X, loss, args):
        log_dict = {}

        optimizer = get_optimizer(self.structure.parameters(), name=args.optimizer, lr=args.lr)

        pbar = tqdm(range(args.num_epochs))

        # outer loop
        for epoch in pbar:
            optimizer.zero_grad()

            # INFO: main calls, note that here all the regularization terms are computed by the respective modules
            alphas, complete_dags, structure_reg = self.structure()

            # initialization for the bilevel part
            sparsifier = self.sparsifier.bl_initialize(complete_dags)
            equations = self.equations.bl_initialize(complete_dags)

            # todo here there's a soso branching, but not ideas how to get it better.
            #  also, can't think of any potential case where the inner part is a full algo but we still might want
            #  a sparsifier.... so for the time being I'm just ignoring the sparsifier
            if self.equations.is_complete_algorithm():
                x_hat, dags, reg = equations(X, complete_dags)
                # TODO maybe call eval here as well also for the complete algorithms
            else:  # GD to optimize the parameters of the equation models and potentially of the sparsifier
                inner_vars = chain(equations.parameters(), sparsifier.parameters())
                # todo also here we'd better decouple the hyperparameters
                inner_opt = get_optimizer(inner_vars, name=args.optimizer, lr=args.lr)
                for inner_iters in range(args.num_inner_iters):
                    inner_opt.zero_grad()

                    dags, sparsifier_reg = sparsifier(complete_dags)
                    x_hat, dags, equations_reg = equations(X, dags)

                    inner_objective = loss(x_hat, X, dim=(1, 2)) + sparsifier_reg + equations_reg
                    inner_objective.sum().backward()

                    inner_opt.step()

                # now evaluate the optimized methods
                equations.eval()
                sparsifier.eval()

                # this now will return the MAP if using L0 STE Bernoulli, not sure what we were doing previously :)
                dags, sparsifier_reg = sparsifier(complete_dags)
                x_hat, dags, equations_reg = equations(X, dags)

            # and now it's done :)
            final_inner_loss = loss(x_hat, X, dim=(1, 2))

            # only final loss should count! to this, we just add the regularization from above
            objective = alphas @ final_inner_loss + structure_reg
            objective.backward()

            optimizer.step()

            # ----- one outer step is done, logging below ------

            pbar.set_description(
                f"objective {objective.item():.2f} | ns {len(complete_dags)} | "
                f"theta norm {self.structure.theta.norm():.4f}"
            )

            log_dict = {
                "epoch": epoch,
                # "number of orderings": num_orderings,
                "objective": objective.item(),
                # "expected loss": exp_loss.item(),
                # "expected l0": exp_l0.item(),
            }
            # TODO should be able to exclude wandb if don't want to use it :)
            wandb.log(log_dict)

            print(self.structure.theta)

        return log_dict

    def _eval(self, X, loss, args):
        # todo THIS PROBABLY WILL BE THE SAME as the joint optimization
        pass


#  ---- old implementation -------------

class DaguerreoOld(nn.Module):
    def __init__(
        self, d, X_torch, smap_init_theta:str, estimator_cls=NNL0Estimator, estimator_kwargs=None
    ):

        super(DaguerreoOld, self).__init__()

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

            # FIXME I think we should have 2 hyperparameters here, one for the
            #  theta L2 norm and one for the equations
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
