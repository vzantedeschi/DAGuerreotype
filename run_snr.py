import logging
import wandb

import numpy as np

from baselines.sortnregress import sortnregress
from daguerreo.args import parse_tuning_args
from daguerreo.data.datasets import get_dataset
from daguerreo.evaluation import evaluate_binary
from daguerreo.utils import get_group_name, get_wandb_mode, init_project_path, init_seeds, log_graph

def run_seed(seed=0):

    init_seeds(seed=seed)

    dag_B, _, X = get_dataset(
        args, to_torch=False, seed=seed + 1
    )

    estimated_B = sortnregress(X)

    log_dict = evaluate_binary(dag_B, estimated_B)

    return log_dict

def run(args, wandb_mode):
    
    group = get_group_name(args)
    
    wandb_run = wandb.init(
        dir=args.results_path,
        entity=args.entity,
        project=args.project,
        name=f"sortnregress",
        group=group,
        config=vars(args),
        reinit=True,
        mode=wandb_mode,
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
                    
                    seed_log_dict = run_seed(seed)
                    log_dict[noise][graph].append(seed_log_dict)
        
        noise_logs = [e for l in log_dict[noise].values() for e in l]
        log_dict[noise]["avg_shdc"] = np.mean([e["shdc"] for e in noise_logs])
        log_dict[noise]["avg_sid"] = np.mean([e["sid"] for e in noise_logs])
        log_dict[noise]["avg_topc"] = np.mean([e["topc"] for e in noise_logs])

    log_dict["avg_shdc"] = np.mean([log_dict[n]["avg_shdc"] for n in args.noise_models])
    log_dict["avg_sid"] = np.mean([log_dict[n]["avg_sid"] for n in args.noise_models])
    log_dict["avg_topc"] = np.mean([log_dict[n]["avg_topc"] for n in args.noise_models])

    wandb.log(log_dict)
    wandb_run.finish()

if __name__ == "__main__":

    argparser = parse_tuning_args()
    args = argparser.parse_args()

    wandb_mode = get_wandb_mode(args)
    save_dir = init_project_path(args=args)

    group = get_group_name(args, log_graph_sem=False)

    run(args, wandb_mode)
