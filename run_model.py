import logging
from pathlib import Path

import torch

import wandb

from args import parse_pipeline_args
from data.datasets import get_dataset
from evaluation import evaluate_binary
from models import Daguerreo
from modules import LARS, LinearL0Estimator
from utils import init_seeds, nll_ev, get_variances, log_graph

def run(args, wandb_mode):

    device = torch.device(args.device)

    project = f"{args.project}"
    project_path = Path(args.results_path) / project
    project_path.mkdir(parents=True, exist_ok=True)

    config = vars(args)

    for seed in range(args.num_seeds):

        init_seeds(seed=seed)
        
        # TODO: use numpy as default and let classes convert to pytorch
        dag_B_torch, dag_W_torch, X_torch = get_dataset(args, to_torch=True, seed=seed+1)

        config["data_seed"] = seed

        name = f"seed={seed}"
        
        try:
            group = f"{args.model}-{args.graph_type}-{args.sem_type}-{args.num_nodes}-{args.num_samples}" # for synthetic data
        except:
            group = f"{args.model}-{args.dataset}-nonlin={args.nonlinear}-std={args.standardize}" # for real data 

        # name_path = project_path / f"{group}-{name}.npy"

        # # if not name_path.is_file():

        wandb_run = wandb.init(
            dir=args.results_path,
            project=project,
            name=name,
            group=group,
            config=config,
            reinit=True,
            mode=wandb_mode,
        )

        log_graph(dag_W_torch.detach().numpy(), "True")

        logging.info(f" Data seed: {seed}, run model {args.model} with {args.estimator} and SMAP's initial theta = {args.smap_init_theta}")
        # log_true_graph(dag_G=dag_W_torch.numpy(), args=args)

        if args.smap_init_theta == "variances":
            smap_init = get_variances(X_torch)
        else:
            smap_init = None

        # TODO: add nonlinear estimator
        if args.estimator == "LARS":
            estimator_cls = LARS
        else:
            estimator_cls = LinearL0Estimator

        model = Daguerreo(args.num_nodes, smap_init=smap_init, estimator_cls=estimator_cls)

        if args.joint:
            logging.info(" Joint optimization")

            log_dict = model.joint_optimization(X_torch, nll_ev, args)
        else:
            logging.info(" Bi-level optimization")

            log_dict = model.bilevel_optimization(X_torch, nll_ev, args)

        model.fit_mode(X_torch, nll_ev, args)

        # evaluate
        estimated_B = model.get_binary_adj().detach().numpy()

        log_graph(estimated_B, "learned")

        log_dict |= evaluate_binary(dag_B_torch.detach().numpy(), estimated_B)


        wandb.log(log_dict)
        logging.info(log_dict)
        wandb_run.finish()


        # print(model.get_graph())
        # np.save(name_path, W_learned) # TODO: save whole model


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
