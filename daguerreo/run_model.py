import logging
import wandb

import numpy as np

from .args import parse_pipeline_args
from .data.datasets import get_dataset
from .evaluation import evaluate_binary
from .models import Daguerro
from .utils import (get_group_name, get_wandb_mode, init_project_path,
                    init_seeds, log_graph, nll_ev, maybe_gpu)


def run(args, wandb_mode, save_dir):
    config = vars(args)

    for seed in range(args.num_seeds):

        config["data_seed"] = seed
        name = f"seed={seed}"
        group = get_group_name(args)

        save_path = save_dir / f"{group}-{name}.npy"

        if save_path.exists():  # do not rerun same experiment again
            continue

        init_seeds(seed=seed)

        # TODO: use numpy as default and let classes convert to pytorch
        #    luca: why? I only see "evaluate_binary" that requires a numpy version....
        dag_B_torch, dag_W_torch, X_torch = get_dataset(
            args, to_torch=True, seed=seed + 1
        )

        wandb_run = wandb.init(
            dir=args.results_path,
            entity=args.entity,
            project=args.project,
            name=name,
            group=group,
            config=config,
            reinit=True,
            mode=wandb_mode,
        )

        log_graph(dag_W_torch.detach().numpy(), "True")

        logging.info(
            f" Data seed: {seed}, run Daguerro with {args.equations} and SMAP's initial theta = {args.smap_init_theta}"
        )

        daguerro = Daguerro.initialize(X_torch, args, args.joint)
        daguerro, X_torch = maybe_gpu(args, daguerro, X_torch)

        log_dict = daguerro(X_torch, nll_ev, args)

        wandb.log(log_dict)
        print(log_dict)

        # todo EVAL part still needs to be done properly
        daguerro.eval()

        _, dags = daguerro(X_torch, nll_ev, args)
        estimated_B = dags[0].cpu().detach().numpy()
        log_graph(estimated_B, "learned")

        log_dict |= evaluate_binary(dag_B_torch.cpu().detach().numpy(), estimated_B)

        wandb.log(log_dict)
        wandb_run.finish()

        logging.info(log_dict)
        np.save(save_path, estimated_B)


if __name__ == "__main__":
    import torch

    torch.set_default_dtype(torch.double)

    argparser = parse_pipeline_args()
    the_args = argparser.parse_args()

    wandb_md = get_wandb_mode(the_args)
    sv_dir = init_project_path(args=the_args)
    run(the_args, wandb_md, sv_dir)
