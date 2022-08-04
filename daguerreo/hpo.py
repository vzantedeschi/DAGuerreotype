import logging
import numpy as np
import optuna
import torch
import wandb

from copy import copy

from .args import parse_tuning_args
from .data.datasets import get_dataset
from .evaluation import evaluate_binary
from .models import Daguerro
from .utils import (get_group_name, get_wandb_mode, init_project_path,
                    init_seeds, log_graph, nll_ev)

class MultiObjectiveHPO():

    def __init__(self, args, project, group, wandb_mode):

        self.original_args = args
        self.project = project
        self.group = group
        self.wandb_mode = wandb_mode

    def _suggest_params(self, trial, args):

        args.pruning_reg = trial.suggest_loguniform("pruning_reg", 1e-6, 1e-1)
        args.l2_theta = trial.suggest_loguniform("l2_theta", 1e-6, 1e-1)
        args.l2_eq = trial.suggest_loguniform("l2_eq", 1e-6, 1e-1)

    def __call__(self, trial):

        args = copy(self.original_args)
        self._suggest_params(trial, args)
       
        wandb_run = wandb.init(
            dir=args.results_path,
            project=self.project,
            name=f"trial_{trial.number}",
            group=self.group,
            config=vars(args),
            reinit=True,
            mode=self.wandb_mode,
        )
        
        losses, shds, l0s, topcs = [], [], [], []
        for seed in range(args.num_seeds):

            init_seeds(seed=seed)

            dag_B_torch, dag_W_torch, X_torch = get_dataset(
                args, to_torch=True, seed=seed + 1
            )

            daguerro = Daguerro.initialize(X_torch, args, args.joint)

            log_dict = daguerro(X_torch, nll_ev, args)

            # todo EVAL part still needs to be done properly
            daguerro.eval()

            x_hat, dags = daguerro(X_torch, nll_ev, args)
            estimated_B = dags[0].detach().numpy()

            log_dict |= evaluate_binary(dag_B_torch.detach().numpy(), estimated_B)

            losses.append(nll_ev(x_hat, X_torch).item())
            shds.append(log_dict["shd"])
            topcs.append(log_dict["topc"])
            l0s.append(log_dict["nnz"])

        avg_loss, avg_l0 = np.mean(losses), np.mean(l0s)
        avg_shd, avg_topc = np.mean(shds), np.mean(topcs)

        if args.wandb:
            wandb.log(log_dict)
            wandb_run.finish()

        return avg_shd, avg_topc

if __name__ == "__main__":

    torch.set_default_dtype(torch.double)

    argparser = parse_tuning_args()
    args = argparser.parse_args()

    wandb_mode = get_wandb_mode(args)
    init_project_path(args=args)

    group = get_group_name(args)

    objective = MultiObjectiveHPO(args, "hpo", group, wandb_mode)
    
    study = optuna.create_study(
            study_name="hpo",
            directions= ["minimize", "maximize"],
            pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=args.num_trials)
