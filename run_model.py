import logging
from pathlib import Path

import numpy as np
import torch

import wandb
from args import parse_pipeline_args
from data.datasets import get_dataset
from init import init_seeds
from train_model import train_and_evaluate

from utils import log_true_graph


def run(args, wandb_mode):

    device = torch.device(args.device)

    project = f"{args.project}"
    project_path = Path(args.results_path) / project
    project_path.mkdir(parents=True, exist_ok=True)

    config = vars(args)

    for seed in range(args.num_seeds):

        init_seeds(seed=seed)
        
        dag_B_torch, dag_W_torch, train_X_torch, val_X_torch = get_dataset(args, to_torch=True, seed=seed+1)

        config["data_seed"] = seed

        name = f"seed={seed}"
        
        try:
            group = f"{args.model}-{args.graph_type}-{args.sem_type}-{args.num_nodes}-{args.num_samples}"
        except:
            group = f"{args.model}-{args.dataset}-lin={args.linear}-std={args.standardize}-{args.l1_reg}"

        name_path = project_path / f"{group}-{name}.npy"

        # if not name_path.is_file():

        wandb_run = wandb.init(
            dir=args.results_path,
            project=project,
            name=name,
            group=group,
            config=vars(args),
            reinit=True,
            mode=wandb_mode,
        )
        logging.info(f"Data seed: {seed}, run model {args.model}")
        log_true_graph(dag_G=dag_W_torch.numpy(), args=args)

        log_dict = {}

        log_dict, W_learned = train_and_evaluate(
            args=args,
            X_torch=train_X_torch,
            B_torch=dag_B_torch,
            log_dict=log_dict,
            val_data=val_X_torch,
        )

        wandb.log(log_dict)
        logging.info(log_dict)
        wandb_run.finish()

        # print(model.get_graph())
        np.save(name_path, W_learned) # TODO: save whole model


if __name__ == "__main__":

    torch.set_default_dtype(torch.double)

    args = parse_pipeline_args()

    if not args.wandb:
        logging.getLogger().setLevel(logging.INFO)
        wandb_mode = "disabled"
    else:
        logging.getLogger().setLevel(logging.WARNING)
        wandb_mode = None

    run(args, wandb_mode)
