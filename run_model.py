import logging

import torch
import wandb

from args import parse_pipeline_args
from data.datasets import get_dataset
from evaluation import evaluate_binary
from models import Daguerreo
from modules import LARS, NNL0Estimator, get_estimator_cls
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
            f" Data seed: {seed}, run model {args.model} with {args.estimator} and SMAP's initial theta = {args.smap_init_theta}"
        )
        # log_true_graph(dag_G=dag_W_torch.numpy(), args=args)

        estimator_cls = get_estimator_cls(args.estimator)

        # [Luca] ideally the main algorithm should accept all valid combinations of
        # the 3 modules + optimization schemes, and that's it :)
        model = Daguerreo(
            d=args.num_nodes,
            X_torch=X_torch,
            smap_init_theta=args.smap_init_theta,
            estimator_cls=estimator_cls,
            estimator_kwargs={
                "linear": not args.nonlinear,
                "hidden": args.hidden,
                "activation": torch.nn.functional.leaky_relu,
            },
        )

        # also this could be handled internally by the full algorithm
        # in the end: if we have one class that represents our algorithm in a framework sense,
        # this class takes 4 arguments + (extra eventually for data)
        # - optimization framework: joint | bilevel
        # - structure learning module: SparseMAP | top-k sparseMAX | ......
        # - sparsification method
        # - structural equations
        if args.joint:
            logging.info(" Joint optimization")

            assert (
                args.estimator != "LARS"
            ), "joint optimization not available for LARS estimator"

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

    argparser = parse_pipeline_args()
    args = argparser.parse_args()

    wandb_mode = get_wandb_mode(args)
    init_project_path(args=args)
    run(args, wandb_mode)
