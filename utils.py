import random
from typing import Iterable
import wandb
import logging

import numpy as np

import torch
from torch.optim import LBFGS, SGD, Adam, AdamW, Optimizer

import networkx as nx
from matplotlib import pyplot as plt

def init_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def get_variances(X):
    return torch.var(X, dim=0, unbiased=False)

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

def plot_DAG(graph: np.ndarray, name: str) -> None:

    G_nx = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    colors = [0.0] * len(graph)

    plt.clf()
    plt.imshow(graph.astype(float), cmap=plt.get_cmap("viridis"))
    plt.colorbar()

    wandb.log({f"adj-{name}": wandb.Image(plt)})

    a_dag = nx.is_directed_acyclic_graph(G_nx)

    logging.info(f"Is {name} graph a DAG? {a_dag}")

    if a_dag:
        logging.info("Topological sort:")

        for l, nodes in enumerate(nx.topological_generations(G_nx)):
            logging.info(nodes)
            for n in nodes:
                colors[n] = 1 - l / len(graph)

    # plot DAG as networkx
    plt.clf()
    nx.draw_networkx(
        G_nx,
        cmap=plt.get_cmap("viridis"),
        node_color=colors,
        with_labels=True,
        font_color="white",
    )

    wandb.log({f"nx-graph-{name}": wandb.Image(plt)})

    return a_dag

def log_graph(dag_G: np.ndarray, name: str) -> None:
    a_dag = plot_DAG(dag_G, name)
    wandb.log({f"{name}_graph": dag_G})

    return a_dag