import logging
import random
from typing import Iterable

import networkx as nx
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.optim import LBFGS, SGD, Adam, AdamW, Optimizer
from pathlib import Path

import causaldag as cd

def get_wandb_mode(args):
    if not args.wandb:
        logging.getLogger().setLevel(logging.INFO)
        wandb_mode = "disabled"
    else:
        logging.getLogger().setLevel(logging.WARNING)
        wandb_mode = None
    return wandb_mode

def get_group_name(args, log_graph_sem=True):
    # experiment naming

    group_name = f"{args.structure}-{args.sparsifier}-{args.equations}-std={args.standardize}"
    try:
        if log_graph_sem:
            group_name += f"-{args.graph_type}-{args.sem_type}-{args.num_nodes}"  # for synthetic data
        else:
            group_name += f"-{args.num_nodes}"  # for synthetic data
    except:
        group_name += f"-{args.dataset}"  # for real data
    return group_name

def init_project_path(args):
    project = f"{args.project}"
    project_path = Path(args.results_path) / project
    project_path.mkdir(parents=True, exist_ok=True)

    return project_path

def init_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def get_variances(X):
    return torch.var(X, dim=0, unbiased=False)


def get_optimizer(params: Iterable, name: str, lr: float) -> Optimizer:

    str_to_optim = {"sgd": SGD, "adam": Adam, "adamW": AdamW}

    try:

        optim = str_to_optim[name]

    except Exception:

        raise NotImplementedError(f"The specified optimizer {name} is not implemented!")

    return optim(params=params, lr=lr)


# --------------------------------------------------------- LOSSES


def nll_ev(output, target, dim=(-2, -1)):
    "negative log likelihood for Daguerro with equal variance"

    loss = (output - target).square()
    loss = loss.sum(dim=dim)

    result = torch.log(loss) * target.shape[-1] / 2
    return result

def squared_loss(output, target, avg_dim=(-2), sum_dim=(-1)):
    "variable-wise squared loss"

    loss = (output - target).square()
    loss = loss.mean(dim=avg_dim) / 2

    result = loss.sum(dim=sum_dim)
    return result

AVAILABLE = {
    'nll_ev': nll_ev,
    'sq': squared_loss,
}

DEFAULT = 'nll_ev'

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


def markov_equiv_class(dag: np.ndarray):
    """

    Args:
        dag: 1 matrix representing the dag

    Returns: a list of matrices representing all the dags in the Markov equivalence class
    """
    # TODO @matt

    cpdag = cd.DAG.from_amat(dag).cpdag()

    all_dags = cpdag.all_dags()
    A = np.zeros((len(all_dags), *dag.shape))
    for i, g in enumerate(all_dags):
        inds = np.array(list(g))
        A[i, inds[:,0], inds[:,1]] = 1

    return A

def greedy_dag(G: np.ndarray):
    G_nx = nx.from_numpy_array(G, create_using=nx.DiGraph)

    sorted_edges = sorted(G_nx.edges(data=True), key=lambda t: t[2].get("weight", 0.0))

    while not nx.is_directed_acyclic_graph(G_nx):

        (u, v, w) = sorted_edges.pop(0)

        if nx.has_path(G_nx, v, u):  # then a cycle exists, try to remove it
            G_nx.remove_edge(u, v)

    return nx.adjacency_matrix(G_nx).todense()

def maybe_gpu(args, *obj):
    """
    Moves objects to cuda if it is enabled and available.


    Args:
        args: arguments namespace
        *obj: any object that have the cuda() method.

    Returns: a list of the same objects moved to cuda, if possible, otherwise returns the original objects

    """
    if not args.nogpu and torch.cuda.is_available():
        return [o.cuda() for o in obj]
        logging.info("Running on gpu")
    return obj


class ApproximateConvergenceChecker():
    def __init__(self, tolerance, min_delta=1.e-4):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.reset()

    def reset(self):
        self.counter = 0
        self.min_prev_loss = torch.inf

    def __call__(self, loss):
        diff_loss = loss - self.min_prev_loss
        if diff_loss < - self.min_delta:
            self.counter = 0
            self.min_prev_loss = loss.item()
        else: self.counter += 1
        if self.counter > self.tolerance:
            return True
        else: return False
