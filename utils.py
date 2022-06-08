import random
from typing import Iterable

import numpy as np

import torch
from torch.optim import LBFGS, SGD, Adam, AdamW, Optimizer

import networkx as nx

def init_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def get_optimizer(params: Iterable, name: str, lr: float) -> Optimizer:
    
    str_to_optim = {
        "sgd": SGD,
        "adam": Adam,
        "adamW": AdamW
    }

    try:

        optim = str_to_optim[name]

    except Exception:

        raise NotImplementedError(
            f"The specified optimizer {name} is not implemented!"
        )

    return optim(params=params, lr=lr)

# --------------------------------------------------------- LOSSES

def nll_ev(output, target, dim=(-2, -1)):
    "negative log likelihood for Daguerro with equal variance"
    
    loss = (output - target).square()
    loss = loss.sum(dim=dim)

    result = torch.log(loss) * target.shape[-1] / 2
    return result

# -------------------------------------------------------- DAG UTILS

def get_topological_rank(graph: np.array) -> np.array:

    g_nx = nx.from_numpy_array(graph, create_using=nx.DiGraph)

    layers = nx.topological_generations(g_nx)

    rank = np.zeros(len(graph))
    for l, layer in enumerate(layers):
        for node in layer:
            rank[node] = l

    return rank