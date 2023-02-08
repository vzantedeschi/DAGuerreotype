import logging

import numpy as np
import optuna
import torch
import wandb

from copy import copy

from .args import parse_tuning_args
from .utils import get_group_name, get_wandb_mode, init_project_path
from .run_model import run_daguerreo

class MultiObjectiveHPO():

    def __init__(self, args, project, group, wandb_mode):

        self.original_args = args
        self.project = project
        self.group = group
        self.wandb_mode = wandb_mode

    def _suggest_params(self, trial, args):

        # args.l2_theta = trial.suggest_loguniform("l2_theta", 1e-6, 1e-1)

        if not args.joint:
            args.lr_theta = trial.suggest_loguniform("lr_theta", 1e-4, 1e-1)

        if args.equations != "lars":
            args.lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
            args.pruning_reg = trial.suggest_loguniform("pruning_reg", 1e-6, 1e-1)
            # args.l2_eq = trial.suggest_loguniform("l2_eq", 1e-6, 1e-1)
        
        if args.equations == "nonlinear":
            args.hidden = trial.suggest_categorical("hidden", [10, 20, 50, 100])

        if args.structure == "tk_sp_max":
            args.smax_max_k = trial.suggest_categorical("smax_max_k", [2, 10, 20, 50, 100])

    def __call__(self, trial):

        args = copy(self.original_args)
        self._suggest_params(trial, args)
       
        wandb_run = wandb.init(
            dir=args.results_path,
            entity=args.entity,
            project=self.project,
            name=f"trial_{trial.number}",
            group=self.group,
            config=vars(args),
            reinit=True,
            mode=self.wandb_mode,
        )
        
        log_dict = {}
        for noise in args.noise_models:

            print(f"Running with noise model \033[1m{noise}\033[0m")
            log_dict[noise] = {}
            args.sem_type = noise

            for graph in args.graph_types:
                
                logging.info(f"graph type \033[1m{graph}\033[0m")
                log_dict[noise][graph] = []
                args.graph_type = graph
                
                for edge_ratio in args.edge_ratios:

                    args.s0 = int(edge_ratio * args.num_nodes)

                    for seed in range(args.num_seeds):
                        
                        try:
                            *_, seed_log_dict = run_daguerreo(args, seed)
                            log_dict[noise][graph].append(seed_log_dict)
                            
                        except RuntimeError as e:
                            logging.error(e)
                            logging.info("Pruning current trial")

                            raise optuna.TrialPruned()
            
            noise_logs = [e for l in log_dict[noise].values() for e in l]
            log_dict[noise]["avg_shdc"] = np.mean([e["shdc"] for e in noise_logs])
            log_dict[noise]["avg_sid"] = np.mean([e["sid"] for e in noise_logs])
            log_dict[noise]["avg_topc"] = np.mean([e["topc"] for e in noise_logs])

        log_dict["avg_shdc"] = np.mean([log_dict[n]["avg_shdc"] for n in args.noise_models])
        log_dict["avg_sid"] = np.mean([log_dict[n]["avg_sid"] for n in args.noise_models])
        log_dict["avg_topc"] = np.mean([log_dict[n]["avg_topc"] for n in args.noise_models])

        wandb.log(log_dict)
        wandb_run.finish()

        return log_dict["avg_shdc"], log_dict["avg_sid"]

if __name__ == "__main__":

    torch.set_default_dtype(torch.double)

    argparser = parse_tuning_args()
    args = argparser.parse_args()

    wandb_mode = get_wandb_mode(args)
    save_dir = init_project_path(args=args)

    group = get_group_name(args, log_graph_sem=False)

    objective = MultiObjectiveHPO(args, args.project, group, wandb_mode)
    
    study = optuna.create_study(
            study_name="hpo",
            directions= ["minimize", "minimize"],
            # pruner=optuna.pruners.MedianPruner() # pruning not supported in MultiObjective
    )

    study.optimize(objective, n_trials=args.num_trials)

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    best_ids = [t.number for t in study.best_trials]
    df_best = df.iloc[best_ids, :] 

    df.to_csv(save_dir / f'{group}-trials.csv')
    df_best.to_csv(save_dir / f'{group}-best-trials.csv')
