from abc import ABCMeta, abstractmethod

from torch import nn

from modules import SparseMapOrdering, LinearL0Estimator

class Daguerreo():

    def __init__(
            self, d, estimator_cls=LinearL0Estimator, smap_tmp=1e-5, smap_init=False, smap_max_iter=100
    ):

        # TODO: allow different init for theta
        self.order_learner = SparseMapOrdering(d, tmp=smap_tmp, init=smap_init, theta_init=None, max_iter=smap_max_iter)

        self.estimator_cls = estimator_cls
        self.d = d

    def bilevel_optimization(self, x, loss, args):

        log_dict = {}

        self.train()

        smap_optim = get_optimizer(self.order_learner.params(), name=args.optimizer, lr=args.lr_theta)  # to update sparsemap parameters

        pbar = tqdm(range(1, args.num_epochs + 1))

        # outer loop
        for epoch in pbar:

            log_dict["epoch"] = epoch

            alphas, orderings = self.order_learner()
            num_orderings = len(alphas)

            # inner problem: learn regressor
            self.estimator = self.estimator_cls(self.d, num_orderings, num_orderings)
            self.estimator.fit(x, orderings, loss, args)

            # evaluate regressor
            self.estimator.eval()

            x_hat = self.estimator(x, orderings)
            fidelity_objective = loss(x_hat, x, dim=(-2, -1))

            # this is the outer loss (i.e. the loss that depends only on the ranking params)
            outer_objective = alphas @ fidelity_objective.detach()
            outer_objective += args.l2_reg * self.order_learner.l2()

            outer_objective.backward()

            smap_optim.step()
            smap_optim.zero_grad()

            pbar.set_description(f"outer objective {outer_objective.item():.2f} | np {len(perms)} | qn {torch.norm(q):.4f}")

            log_dict["number of orderings"] = num_orderings

            wandb.log(log_dict)

        # TODO: call SparseMAP instead, but with very low temperature to get the MAP
        mode_ordering_id = torch.argmax(alphas)
        self.mode_ordering = orderings[mode_ordering_id]

    def joint_optimization(self, x, loss, args):
        pass

    def fit_mode(self, x, loss, args):

        self.mode_estimator = self.estimator_cls(self.d, 1, 1) # one structure, one set of equations
        self.mode_estimator.fit(x, self.mode_ordering, loss, args)