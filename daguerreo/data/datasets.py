import math

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .dag_simulation import sample_data, simulate_dag


def get_synthetic_dataset(args, *a):
    dag_B = simulate_dag(
        d=args.num_nodes, s0=args.s0, graph_type=args.graph_type
    )  # binary adj matrix

    dag_W, data = sample_data(
        B=dag_B,
        num_samples=args.num_samples,
        sem_type=args.sem_type,
        noise_scale=args.noise_scale,
    )

    return dag_B, dag_W, data


def get_sachs_dataset(args_ns, *a):

    data = np.load("./data/sachs/continuous/data1.npy")
    dag_B = np.load("./data/sachs/continuous/DAG1.npy")

    return dag_B, dag_B, data


def get_syntren_dataset(args_ns, seed=1):

    seed = np.clip(seed, 1, 10)
    data = np.load(f"./data/syntren/data{seed}.npy")
    dag_B = np.load(f"./data/syntren/DAG{seed}.npy")

    return dag_B, dag_B, data


def get_dataset(args, to_torch=True, seed=1):
    # TODO: load cpdag as well
    # TODO: synthetic - return Estimator
    datasets = {
        "synthetic": get_synthetic_dataset,
        "sachs": get_sachs_dataset,
        "syntren": get_syntren_dataset,
    }

    dag_B, dag_W, data = datasets[args.dataset](args, seed)

    args.num_nodes = len(dag_B)
    args.num_samples = len(data)

    # standardize or only recenter data
    scaler = StandardScaler(with_std=args.standardize)
    data = scaler.fit_transform(data)

    if to_torch:
        dag_B, dag_W, data = (
            torch.from_numpy(dag_B).to(args.device),
            torch.from_numpy(dag_W).to(args.device),
            torch.from_numpy(data).to(args.device),
        )

    return dag_B, dag_W, data
