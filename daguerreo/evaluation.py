import logging

import numpy as np

import causaldag as cd

from cdt.metrics import SID, SHD_CPDAG
from .utils import get_topological_rank

# -------------------------------------------------------------------------- METRICS


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """

    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    
    B_cpdag_true = cd.DAG.from_amat(B_true).cpdag()
    B_cpdag_est = cd.DAG.from_amat(B_est).cpdag()
    
    return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "nnz": pred_size, "shdc": B_cpdag_true.shd(B_cpdag_est)}


def topological_rank_corr(B_true: np.ndarray, B_est: np.ndarray) -> dict:

    true_rank = get_topological_rank(B_true)
    est_rank = get_topological_rank(B_est)

    coeff = np.corrcoef(true_rank, est_rank)[0, 1]

    if np.isnan(coeff):
        coeff = 0.0

    return coeff

# dtop metric from https://arxiv.org/abs/2203.04413
def topological_order_divergence(B_true: np.ndarray, order_est: np.ndarray) -> dict:

    d = len(B_true)

    # get mask of edges that are not consistent with estimated ordering
    M = np.tril(np.ones((d, d)), k=1)
    mask = M[order_est[..., None], order_est[:, None]]

    # count all true edges that are not consistent with estimated ordering
    coeff = (B_true * mask).sum()

    return coeff

def eval_order(B_true: np.ndarray, B_est: np.ndarray, order_est: np.ndarray=None):

    res = {"topc": topological_rank_corr(B_true, B_est)}

    if order_est is not None:
        res |= {"dtop": topological_order_divergence(B_true, order_est)}

    return res

# ------------------------------------------------------------------ DAG EVALUATION


def evaluate_binary(true_B: np.array, estimated_B: np.array, estimated_order: np.array=None):
    try:
        res_dict = {"sid": SID(true_B, estimated_B).item()}
    except FileNotFoundError:
        logging.warning('SID not computable, R might be missing')
        res_dict = {"sid": None}
    # res_dict = {}
    res_dict |= count_accuracy(B_true=true_B, B_est=estimated_B)
    res_dict |= eval_order(true_B, estimated_B)

    return res_dict
