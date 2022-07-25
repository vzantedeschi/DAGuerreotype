"""Trying to find an efficient top-k algorithm for permutations"""

import numpy as np
from itertools import permutations

import torch
from entmax import sparsemax

from ._kbest import KBestRankings
from .sparsemap import sparsemap_rank


def dot_perm(x, pi):
    """compute <x, pi>

    x: torch Tensor, shape [d]
    pi: torch Tensor[long], shape [k, d]
    """

    pi_ = pi.to(dtype=x.dtype) + 1
    return pi_ @ x


def sparsemax_rank(x, max_k=100, prune_output=True):
    """top-k sparsemax on the ranking problem (permutahedron).

    Parameters
    ----------

    x: torch.Tensor shape [d]
        The vector of scores to be ranked.

    max_k: int, default: 100
        The maximum value of top-ranked permutations to compute.

    prune_output: bool, default: True
        Whether to prune the permutations with zero probabilities.
        If False, the output always has shape ([max_k], [max_k, d]).
        If True, the output can have fewer rows.


    Returns
    -------

    probas: torch.Tensor,
        sparsemax probabilities, differentiable wrt the score vector.

    rankings: torch.Tensor
        discrete rankings of the input vector, ordered by how close
        they are to sorting the vector ascendingly.
        rankings[0] is always the inverse permutation of argsort(x).
    """
    ranker = KBestRankings(x.to(device='cpu', dtype=torch.double), max_k)
    rankings, _ = ranker.compute(max_k)

    # recompute scores on clean autodiffable path from x
    scores = dot_perm(x, rankings)
    probas = sparsemax(scores)

    if prune_output:
        mask = probas > 0
        return probas[mask], rankings[mask]
    else:
        return probas, rankings
