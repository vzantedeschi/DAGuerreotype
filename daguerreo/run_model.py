import logging
import wandb

import numpy as np

from .args import parse_pipeline_args
from .data.datasets import get_dataset
from .evaluation import evaluate_binary
from .models import Daguerro
from . import utils

def run_daguerreo(args, seed=0):

    utils.init_seeds(seed=seed)

    dag_B_torch, dag_W_torch, X_torch = get_dataset(
        args, to_torch=True, seed=seed + 1
    )

    daguerro = Daguerro.initialize(X_torch, args, args.joint)
    daguerro, X_torch = utils.maybe_gpu(args, daguerro, X_torch)

    log_dict = daguerro(X_torch, utils.AVAILABLE[args.loss], args)

    # todo EVAL part still needs to be done properly
    daguerro.eval()

    _, dags = daguerro(X_torch, utils.AVAILABLE[args.loss], args)

    estimated_B = dags[0].detach().cpu().numpy()

    log_dict |= evaluate_binary(dag_B_torch.detach().numpy(), estimated_B)

    return dag_W_torch.detach().numpy(), estimated_B, log_dict

def run(args, wandb_mode, save_dir):
    
    config = vars(args)
    group = utils.get_group_name(args)

    for seed in range(args.num_seeds):

        config["data_seed"] = seed
        name = f"seed={seed}"

        save_path = save_dir / f"{group}-{name}.npy"

        if save_path.exists():  # do not rerun same experiment again
            continue

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
        
        logging.info(
            f" Data seed: {seed}, run DAGuerreotype with {args.structure}, {args.equations} edge estimator and initial score vector theta = {args.init_theta}"
        )
        
        if args.model == "daguerreo":
            true_W, estimated_B, log_dict = run_daguerreo(args, seed)
        elif args.model == "snr":
            true_W, estimated_B, log_dict = run_snr(args, seed)

        utils.log_graph(true_W, "True")
        utils.log_graph(estimated_B, "learned")

        wandb.log(log_dict)
        wandb_run.finish()

        print(log_dict)
        np.save(save_path, estimated_B)

if __name__ == "__main__":
    import torch

    torch.set_default_dtype(torch.double)

    argparser = parse_pipeline_args()
    the_args = argparser.parse_args()

    wandb_md = utils.get_wandb_mode(the_args)
    sv_dir = utils.init_project_path(args=the_args)
    run(the_args, wandb_md, sv_dir)
