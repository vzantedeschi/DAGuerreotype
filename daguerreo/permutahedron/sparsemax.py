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

    # rho = torch.arange(x.shape[0]).to(dtype=x.dtype, device=x.device) + 1
    # return x[pi] @ rho
    pi_ = pi.to(dtype=x.dtype) + 1
    return pi_ @ x


def sparsemax_rank(x, max_k=100, prune_output=True):
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


def main():
    x = torch.randn(5).double()
    print("sparsemax")
    print(sparsemax_rank(x, max_k=4))

    print("sparsemap")
    print(sparsemap_rank(x, max_iter=100, init=False))


if __name__ == '__main__':
    main()

