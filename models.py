import torch
import wandb
from torch import nn
from tqdm import tqdm

from typing import Type, Optional

# from modules import Structure, Sparsifier, Equations
import equations
import sparsifiers
import structures
import utils
from equations import Equations
from structures import Structure
from sparsifiers import Sparsifier
from utils import get_optimizer, get_variances
from itertools import chain


# TODO do we need this to subclass nn.Module?
class Daguerro(torch.nn.Module):

    def __init__(self, structure: Structure,
                 sparsifier: Sparsifier,
                 equation: Equations) -> None:
        super(Daguerro, self).__init__()
        self.structure = structure
        self.sparsifier = sparsifier
        self.equation = equation

    @staticmethod
    def initialize(X, args, joint=False):
        dag_cls = DaguerroJoint if joint else DaguerroBilevel
        # INFO: initialization is deferred to the specific methods to deal with all the modules'
        # hyperparameters inplace
        structure = structures.AVAILABLE[args.structure].initialize(X, args)
        sparsifier = sparsifiers.AVAILABLE[args.sparsifier].initialize(X, args, joint)
        equation = equations.AVAILABLE[args.equations].initialize(X, args, joint)
        return dag_cls(structure, sparsifier, equation)

    def forward(self, X, loss, args):
        # TODO remove args from here and push everything to the initialize method,
        #  the rationale is that in this way one can also run the algorithm not form the CLI!
        #  as it stands it is very complicated to do that!
        if self.training:
            return self._learn(X, loss, args)
        else:
            return self._eval(X, loss, args)

    def _learn(self, X, loss, args): raise NotImplementedError()

    def _eval(self, X, loss, args):
        # FIXME put something meaningful here, i.e. fitting the mode, if we want to go that way...
        alphas, complete_dags, structure_reg = self.structure()
        dags, sparsifier_reg = self.sparsifier(complete_dags)
        x_hat, dags, equations_reg = self.equation(X, dags)
        return x_hat, dags


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
            x_hat, dags, equations_reg = self.equation(X, dags)

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

            # print(self.structure.theta)

        return log_dict


class DaguerroBilevel(Daguerro):

    def _learn(self, X, loss, args):
        log_dict = {}

        optimizer = get_optimizer(self.structure.parameters(), name=args.optimizer, lr=args.lr_theta)

        pbar = tqdm(range(args.num_epochs))

        # outer loop
        for epoch in pbar:
            optimizer.zero_grad()

            # INFO: main calls, note that here all the regularization terms are computed by the respective modules
            alphas, complete_dags, structure_reg = self.structure()

            # fit the equations and the sparsifier, if any
            self.equation.fit(X, complete_dags, self.sparsifier, loss)

            # now evaluate the optimized methods
            self.equation.eval()
            self.sparsifier.eval()

            # this now will return the MAP (mode) if e.g. using L0 STE Bernoulli,
            #   todo @vale not sure what we were doing previously :)
            dags, sparsifier_reg = self.sparsifier(complete_dags)
            x_hat, dags, equations_reg = self.equation(X, dags)

            # and now it's done :) don't think it's meaningfully to consider the (inner) regularizers here
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
                "objective": final_inner_loss[0].item(),
                # "expected loss": exp_loss.item(),
                # "expected l0": exp_l0.item(),
            }

            # some printing tbd
            # if epoch % 100 == 0:
            #     print(self.structure.theta)
            #     print(self.structure.theta.grad)
            #     print(alphas)
            #     print(dags[0])
            #     print(utils.get_topological_rank(dags[0].detach().numpy()))
            #     pass

        return log_dict

# ---- old implementation of eval part
#
#     def fit_mode(self, x, loss, args):
#
#         # todo: use sort instead
#         (
#             alphas,
#             self.mode_masking,
#         ) = self.smap_masking()  # with default values, returns MAP
#         assert len(alphas) == 1
#
#         self.mode_estimator = self.estimator_cls(
#             self.d, 1, **self.estimator_kwargs
#         )  # one structure
#         self.mode_estimator.fit(x, self.mode_masking, loss, args)
#
#     def get_binary_adj(self):
#
#         return self.mode_estimator.get_structure(self.mode_masking).squeeze(0)
