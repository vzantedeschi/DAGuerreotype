import logging

import torch
import wandb

from args import parse_pipeline_args
from data.datasets import get_dataset
from evaluation import evaluate_binary
from models import Daguerro
from utils import (get_group_name, get_wandb_mode, init_project_path,
                   init_seeds, log_graph, nll_ev)


def run(args, wandb_mode):

    config = vars(args)

    for seed in range(args.num_seeds):

        init_seeds(seed=seed)

        # TODO: use numpy as default and let classes convert to pytorch
        dag_B_torch, dag_W_torch, X_torch = get_dataset(
            args, to_torch=True, seed=seed + 1
        )

        config["data_seed"] = seed

        name = f"seed={seed}"

        # name_path = project_path / f"{group}-{name}.npy"
        # # if not name_path.is_file():

        wandb_run = wandb.init(
            dir=args.results_path,
            project=args.project,
            name=name,
            group=get_group_name(args),
            config=config,
            reinit=True,
            mode=wandb_mode,
        )

        log_graph(dag_W_torch.detach().numpy(), "True")

        logging.info(
            f" Data seed: {seed}, run model {args.model} with {args.equations} and SMAP's initial theta = {args.smap_init_theta}"
        )
        # log_true_graph(dag_G=dag_W_torch.numpy(), args=args)

        # estimator_cls = get_estimator_cls(args.estimator)
        daguerro = Daguerro.initialize(X_torch, args, args.joint)

        log_dict = daguerro(X_torch, nll_ev, args)

        if args.wandb:
            wandb.log(log_dict)
        print(log_dict)

        # THIS IS STILL TO BE DONE
        daguerro.eval()

        x_hat, dags = daguerro(X_torch, nll_ev, args)
        estimated_B = dags[0].detach().numpy()
        log_graph(estimated_B, "learned")

        log_dict |= evaluate_binary(dag_B_torch.detach().numpy(), estimated_B)

        if args.wandb:
            wandb.log(log_dict)
            wandb_run.finish()

        logging.info(log_dict)


        # --- todo eval part below is to be done!~
        # model.fit_mode(X_torch, nll_ev, args)
        #
        # # evaluate
        # estimated_B = model.get_binary_adj().detach().numpy()
        #
        # log_graph(estimated_B, "learned")
        #
        # log_dict |= evaluate_binary(dag_B_torch.detach().numpy(), estimated_B)
        #
        # wandb.log(log_dict)
        # logging.info(log_dict)
        # wandb_run.finish()


if __name__ == "__main__":

    torch.set_default_dtype(torch.double)

    argparser = parse_pipeline_args()
    args = argparser.parse_args()

    wandb_mode = get_wandb_mode(args)
    init_project_path(args=args)
    run(args, wandb_mode)
