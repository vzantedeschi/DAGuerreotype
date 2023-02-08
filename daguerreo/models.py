import logging
from copy import copy

import wandb

import torch
from tqdm import tqdm

from . import equations
from . import sparsifiers
from . import structures
from . import utils
from .equations import Equations
from .structures import Structure
from .sparsifiers import Sparsifier
from .utils import get_optimizer, get_topological_order

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

        if args.structure == "true_rank":
            ordering = torch.from_numpy(get_topological_order(G))
            dag_cls = FixedStructureDaguerro
            joint = False # force bilevel optim

        elif args.structure == "rnd_rank":
            ordering = torch.randperm(X.shape[1])
            dag_cls = FixedStructureDaguerro
            joint = False # force bilevel optim

        else:
            dag_cls = DaguerroJoint if joint else DaguerroBilevel
            ordering = None

        # INFO: initialization is deferred to the specific methods to deal with all the modules'
        # hyperparameters inplace
        structure = structures.AVAILABLE[args.structure].initialize(X, args, ordering)
        sparsifier = sparsifiers.AVAILABLE[args.sparsifier].initialize(X, args, joint)
        equation = equations.AVAILABLE[args.equations].initialize(X, args, joint)
        return dag_cls(structure, sparsifier, equation)

    def forward(self, X, loss, args):
        if self.training:
            return self._learn(X, loss, args)
        else:
            return self._eval(X, loss, args)

    def _learn(self, X, loss, args): raise NotImplementedError()

    def _eval(self, X, loss, args):
        _, complete_dags, _ = self.structure()

        logging.info(f'Fitting the mode. We have {len(complete_dags)} complete dags!')

        # -- increase the fitting time (TODO make this passage a bit more decent)
        ex_args = copy(args)
        ex_args.num_inner_iters *= 5
        ex_args.es_tol_inner *= 5

        self.equation = self.equation.initialize(X, ex_args)
        self.sparsifier = self.sparsifier.initialize(X, ex_args)

        self.equation.fit(X, complete_dags, self.sparsifier, loss)
        logging.info('Done fitting the mode!')

        self.sparsifier.eval()
        self.equation.eval()

        dags, _ = self.sparsifier(complete_dags)
        x_hat, dags, _ = self.equation(X, dags)
        return x_hat, dags


class DaguerroJoint(Daguerro):

    def _learn(self, X, loss, args):
        # here all should be initialized
        log_dict = {}

        optimizer = get_optimizer(self.parameters(), name=args.optimizer, lr=args.lr)
        convergence_checker = utils.ApproximateConvergenceChecker(args.es_tol, args.es_delta)

        pbar = tqdm(range(args.num_epochs))

        # outer loop
        for epoch in pbar:

            optimizer.zero_grad()

            # INFO: main calls, note that here all the regularization terms are computed by the respective modules
            alphas, complete_dags, structure_reg = self.structure()
            dags, sparsifier_reg = self.sparsifier(complete_dags)
            x_hat, dags, equations_reg = self.equation(X, dags)

            rec_loss = loss(x_hat, X)

            objective = alphas @ (rec_loss + sparsifier_reg + equations_reg) + structure_reg
            objective.backward()

            optimizer.step()

            pbar.set_description(
                f"objective {objective.item():.2f} | ns {len(complete_dags)} | "
                f"theta norm {self.structure.theta.norm():.4f}"
            )

            log_dict = {
                "epoch": epoch,
                "number of orderings": len(alphas),
                "objective": objective.item(),
            }

            wandb.log(log_dict)
            if convergence_checker(objective):
                logging.info(f'Objective approx convergence at epoch {epoch}')
                break

        return log_dict


class DaguerroBilevel(Daguerro):

    def _learn(self, X, loss, args):
        log_dict = {}

        optimizer = get_optimizer(self.structure.parameters(), name=args.optimizer, lr=args.lr_theta)
        convergence_checker = utils.ApproximateConvergenceChecker(args.es_tol, args.es_delta)

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
            dags, _ = self.sparsifier(complete_dags)
            x_hat, dags, _ = self.equation(X, dags)

            # and now it's done
            final_inner_loss = loss(x_hat, X)

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
                "number of orderings": len(alphas),
                "objective": objective.item(),
            }
            wandb.log(log_dict)

            if convergence_checker(objective):
                logging.info(f'Outer obj. approx convergence at epoch {epoch}')
                break

        return log_dict

class FixedStructureDaguerro(Daguerro):
    
    def _learn(self, X, loss, args, metric_fun=None):
        log_dict = {}

        # INFO: main calls, note that here all the regularization terms are computed by the respective modules
        alphas, complete_dags, _ = self.structure()

        # fit the equations and the sparsifier, if any
        self.equation.fit(X, complete_dags, self.sparsifier, loss)

        # now evaluate the optimized methods
        self.equation.eval()
        self.sparsifier.eval()

        # this now will return the MAP (mode) if e.g. using L0 STE Bernoulli,
        dags, _ = self.sparsifier(complete_dags)
        x_hat, dags, _ = self.equation(X, dags)

        final_inner_loss = loss(x_hat, X)
        objective = alphas @ final_inner_loss

        log_dict = {
            "number of orderings": len(alphas),
            "objective": objective.item(),
        }

        wandb.log(log_dict)
        
        return log_dict
    
    def _eval(self, X, loss, args):
        _, complete_dags, _ = self.structure()

        self.sparsifier.eval()
        self.equation.eval()

        dags, _ = self.sparsifier(complete_dags)
        x_hat, dags, _ = self.equation(X, dags)

        return x_hat, dags