"""
Adapted from
https://raw.githubusercontent.com/xunzheng/notears/ba61337bd0e5410c04cc708be57affc191a8c424/notears/utils.py

"""
from typing import Tuple

import igraph as ig
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.gaussian_process import GaussianProcessRegressor


def is_dag(W: np.ndarray) -> bool:
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d: int, s0: int, graph_type: str) -> np.ndarray:
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)

        G = ig.Graph.Random_Bipartite(
            top, d - top, m=min(s0, top * (d - top)), directed=True, neimode=ig.OUT
        )
        B = _graph_to_adjmat(G)
    else:
        raise ValueError("unknown graph type:", graph_type)
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(
    B: np.ndarray,
    w_ranges: "tuple[tuple[float, float], tuple[float, float]]" = (
        (-2, -0.5),
        (0.5, 2),
    ),
) -> np.ndarray:
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(
    W: np.ndarray, num_samples: int, sem_type: str, noise_scale: np.ndarray = 1
) -> np.ndarray:
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        num_samples (int): num of samples, num_samples=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type in ["gauss", "gauss-heter"]:
            z = np.random.normal(scale=scale, size=num_samples)
            x = X @ w + z
        elif sem_type == "exp":
            z = np.random.exponential(scale=scale, size=num_samples)
            x = X @ w + z
        elif sem_type == "gumbel":
            z = np.random.gumbel(scale=scale, size=num_samples)
            x = X @ w + z
        elif sem_type == "uniform":
            z = np.random.uniform(low=-scale, high=scale, size=num_samples)
            x = X @ w + z
        elif sem_type == "logistic":
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == "poisson":
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError("unknown sem type:", sem_type)
        return x

    d = W.shape[0]

    if np.isscalar(noise_scale):

        if sem_type == "gauss-heter":
            scale_vec = np.random.uniform(low=0, high=noise_scale, size=d)
        else:
            scale_vec = noise_scale * np.ones(d)

    else:
        if len(noise_scale) != d:
            raise ValueError("noise scale must be a scalar or have length d")
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError("W must be a DAG")
    if np.isinf(num_samples):  # population risk for linear gauss SEM
        if sem_type == "gauss":
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError("population risk not available")
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([num_samples, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(
    B: np.ndarray, num_samples: int, sem_type: str, noise_scale: np.ndarray = 1
) -> np.ndarray:
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        num_samples (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.array): scale parameter of additive noise, default one

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """

    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=num_samples)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == "mlp":
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == "mim":
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == "gp":
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == "gp-add":
            gp = GaussianProcessRegressor()
            x = (
                sum(
                    [
                        gp.sample_y(X[:, i, None], random_state=None).flatten()
                        for i in range(X.shape[1])
                    ]
                )
                + z
            )
        else:
            raise ValueError("unknown sem type")
        return x

    d = B.shape[0]

    if np.isscalar(noise_scale):

        scale_vec = noise_scale * np.ones(d)

    else:
        if len(noise_scale) != d:
            raise ValueError("noise scale must be a scalar or have length d")
        scale_vec = noise_scale

    X = np.zeros([num_samples, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def sample_data(
    B: np.ndarray, num_samples: int, sem_type: str, noise_scale: float
) -> Tuple[np.ndarray, np.ndarray]:

    if sem_type in [
        "mlp",
        "mim",
        "gp",
        "gp-add",
    ]:  # non-linear Structural Equation Model

        W = B

        data = simulate_nonlinear_sem(
            B=B,
            num_samples=num_samples,
            sem_type=sem_type,
            noise_scale=noise_scale,
        )

    else:  # linear Structural Equation Model
        W = simulate_parameter(
            B
        )  # generate a weighted adj matrix from binary one (with positive and negative weights)

        data = simulate_linear_sem(
            W=W,
            num_samples=num_samples,
            sem_type=sem_type,
            noise_scale=noise_scale,
        )

    return W, data
