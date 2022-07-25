"""Trying to find an efficient top-k algorithm for permutations"""

import numpy as np
from itertools import permutations

import torch
from ._kbest import KBestRankings
from .sparsemap import sparsemap_rank


def sparsemax_rank(x, max_k=100):
    x_ = x.cpu()
    ranker = KBestRankings(x_, max_k)
    rankings, scores = ranker.compute(max_k)

    scores_ = rankings.to(dtype=x_.dtype) @ x_

    print(rankings)
    print(scores)
    print(scores_)

    return


def main():
    x = torch.randn(5).double()
    print("sparsemax")
    print(sparsemax_rank(x, max_k=4))

    print("sparsemap")
    print(sparsemap_rank(x, max_iter=100, init=False))

if __name__ == '__main__':
    main()

