import numpy as np
import torch

from lpsmap.ad3qp.factor_graph import PFactorGraph

from ._sparsemap import Permutahedron


class _BaseSparseMAP(torch.autograd.Function):
    @classmethod
    def run_sparsemap(cls, ctx, x):
        ctx.n = x.shape[0]
        ctx.fg = PFactorGraph()
        ctx.fg.set_verbosity(1)
        ctx.variables = [ctx.fg.create_binary_variable() for _ in range(ctx.n)]
        ctx.f = cls.make_factor(ctx)
        ctx.fg.declare_factor(ctx.f, ctx.variables)
        x_np = x.detach().cpu().numpy().astype(np.double)
        # if true, random-initialize the algorithm
        if ctx.init:
            init = torch.rand(x_np.shape[0], dtype=torch.double)
            ctx.f.init_active_set_from_scores(init, [])

        _, _ = ctx.f.solve_qp(x_np, [], max_iter=ctx.max_iter)
        aset, p = ctx.f.get_sparse_solution()

        p = p[: len(aset)]

        aset = torch.tensor(aset, dtype=torch.long, device=x.device) - 1
        p = torch.tensor(p, dtype=torch.double, device=x.device)
        ctx.mark_non_differentiable(aset)
        return p, aset

    @classmethod
    def jv(cls, ctx, dp):
        # d_eta_u = np.empty(ctx.n, dtype=np.double)
        # d_eta_v = np.empty(0, dtype=np.double)
        # ctx.f.dist_jacobian_vec(dp.cpu().numpy().astype(np.double), d_eta_u, d_eta_v)
        # d_eta_u = torch.from_numpy(d_eta_u, dtype=dp.dtype, device=dp.device)
        # return d_eta_u
        dp_npy = dp.cpu().numpy()
        d_eta_u = np.empty(ctx.n, dtype=dp_npy.dtype)
        d_eta_v = np.empty(0, dtype=dp_npy.dtype)
        ctx.f.dist_jacobian_vec(dp.cpu().numpy(), d_eta_u, d_eta_v)
        d_eta_u = torch.from_numpy(d_eta_u).to(dtype=dp.dtype, device=dp.device)
        return d_eta_u


class RankSparseMAP(_BaseSparseMAP):
    @classmethod
    def make_factor(cls, ctx):
        f = Permutahedron()
        f.initialize(ctx.n)
        return f

    @classmethod
    def forward(cls, ctx, x, max_iter, init):
        ctx.max_iter = max_iter
        ctx.init = init
        return cls.run_sparsemap(ctx, x)

    @classmethod
    def backward(cls, ctx, dp, daset):
        res = cls.jv(ctx, dp.to(dtype=torch.float64, device="cpu"))
        return res.to(dtype=dp.dtype, device=dp.device).unsqueeze(-1), None, None
        # return cls.jv(ctx, dp), None, None, None


def sparsemap_rank(x, max_iter=100, init=True):
    return RankSparseMAP.apply(x, max_iter, init)
