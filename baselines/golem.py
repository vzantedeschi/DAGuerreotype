import logging

import numpy as np
import tensorflow as tf


class GolemModel:
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """

    def __init__(
        self, n, d, lambda_1, lambda_2, equal_variances=True, seed=1, B_init=None
    ):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelibood objective. Default: True.
            seed (int): Random seed. Default: 1.
            B_init (numpy.ndarray or None): [d, d] weighted matrix for
                initialization. Set to None to disable. Default: None.
        """
        self.n = n
        self.d = d
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.B_init = B_init

        self._build()
        self._init_session()

    def _init_session(self):
        """Initialize tensorflow session."""
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
            )
        )

    def _build(self):
        """Build tensorflow graph."""
        tf.compat.v1.reset_default_graph()

        # Placeholders and variables
        self.lr = tf.compat.v1.placeholder(tf.float32)
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[self.n, self.d])
        self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        if self.B_init is not None:
            self.B = tf.Variable(tf.convert_to_tensor(self.B_init, tf.float32))
        else:
            self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        self.B = self._preprocess(self.B)

        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = (
            self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h
        )

        # Optimizer
        self.train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.lr
        ).minimize(self.score)
        logging.debug("Finished building tensorflow graph.")

    def _preprocess(self, B):
        """Set the diagonals of B to zero.

        Args:
            B (tf.Tensor): [d, d] weighted matrix.

        Returns:
            tf.Tensor: [d, d] weighted matrix.
        """
        return tf.linalg.set_diag(B, tf.zeros(B.shape[0], dtype=tf.float32))

    def _compute_likelihood(self):
        """Compute (negative log) likelihood in the linear Gaussian case.

        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        if self.equal_variances:  # Assuming equal noise variances
            return (
                0.5
                * self.d
                * tf.math.log(tf.square(tf.linalg.norm(self.X - self.X @ self.B)))
                - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]
            )
        else:  # Assuming non-equal noise variances
            return (
                0.5
                * tf.math.reduce_sum(
                    tf.math.log(
                        tf.math.reduce_sum(tf.square(self.X - self.X @ self.B), axis=0)
                    )
                )
                - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]
            )

    def _compute_L1_penalty(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return tf.norm(self.B, ord=1)

    def _compute_h(self):
        """Compute DAG penalty.

        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return tf.linalg.trace(tf.linalg.expm(self.B * self.B)) - self.d


class GolemTrainer:
    """Set up the trainer to solve the unconstrained optimization problem of GOLEM."""

    def __init__(self, learning_rate=1e-3):
        """Initialize self.

        Args:
            learning_rate (float): Learning rate of Adam optimizer.
                Default: 1e-3.
        """
        self.learning_rate = learning_rate

    def train(
        self,
        model,
        X,
        num_iter,
        interventional_data=None,
        checkpoint_iter=None,
        output_dir=None,
    ):
        """Training and checkpointing.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.
            num_iter (int): Number of iterations for training.
            checkpoint_iter (int): Number of iterations between each checkpoint.
                Set to None to disable. Default: None.
            output_dir (str): Output directory to save training outputs. Default: None.

        Returns:
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        model.sess.run(tf.compat.v1.global_variables_initializer())

        logging.info("Started training for {} iterations.".format(num_iter))
        for i in range(0, int(num_iter) + 1):
            if i == 0:  # Do not train here, only perform evaluation
                score, likelihood, h, B_est = self.eval_iter(model, X)
            else:  # Train
                score, likelihood, h, B_est = self.train_iter(model, X)
                if interventional_data is not None:
                    score, likelihood, h, B_est = self.train_iter(
                        model, interventional_data
                    )

            if checkpoint_iter is not None and i % checkpoint_iter == 0:
                self.train_checkpoint(i, score, likelihood, h, B_est, output_dir)

        return B_est

    def eval_iter(self, model, X):
        """Evaluation for one iteration. Do not train here.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        score, likelihood, h, B_est = model.sess.run(
            [model.score, model.likelihood, model.h, model.B],
            feed_dict={model.X: X, model.lr: self.learning_rate},
        )

        return score, likelihood, h, B_est

    def train_iter(self, model, X):
        """Training for one iteration.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        _, score, likelihood, h, B_est = model.sess.run(
            [model.train_op, model.score, model.likelihood, model.h, model.B],
            feed_dict={model.X: X, model.lr: self.learning_rate},
        )

        return score, likelihood, h, B_est

    def train_checkpoint(self, i, score, likelihood, h, B_est, output_dir):
        """Log and save intermediate results/outputs.

        Args:
            i (int): i-th iteration of training.
            score (float): value of score function.
            likelihood (float): value of likelihood function.
            h (float): value of DAG penalty.
            B_est (numpy.ndarray): [d, d] estimated weighted matrix.
            output_dir (str): Output directory to save training outputs.
        """
        logging.info(
            "[Iter {}] score {:.3E}, likelihood {:.3E}, h {:.3E}".format(
                i, score, likelihood, h
            )
        )

        if output_dir is not None:
            # Save the weighted matrix (without post-processing)
            create_dir("{}/checkpoints".format(output_dir))
            np.save("{}/checkpoints/B_iteration_{}.npy".format(output_dir, i), B_est)


def golem(
    X,
    lambda_1,
    lambda_2,
    w_threshold=0.3,
    equal_variances=True,
    num_iter=1e5,
    learning_rate=1e-3,
    seed=1,
    checkpoint_iter=None,
    output_dir=None,
    B_init=None,
    interventional_data=None,
):
    """Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.
    Args:
        X (numpy.ndarray): [n, d] data matrix.
        lambda_1 (float): Coefficient of L1 penalty.
        lambda_2 (float): Coefficient of DAG penalty.
        equal_variances (bool): Whether to assume equal noise variances
            for likelibood objective. Default: True.
        num_iter (int): Number of iterations for training.
        learning_rate (float): Learning rate of Adam optimizer. Default: 1e-3.
        seed (int): Random seed. Default: 1.
        checkpoint_iter (int): Number of iterations between each checkpoint.
            Set to None to disable. Default: None.
        output_dir (str): Output directory to save training outputs.
        B_init (numpy.ndarray or None): [d, d] weighted matrix for initialization.
            Set to None to disable. Default: None.
    Returns:
        numpy.ndarray: [d, d] estimated weighted matrix.
    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """

    tf.compat.v1.disable_eager_execution()

    # Set up model
    logging.info("Find solution with equal variance formulation.")
    n, d = X.shape
    model = GolemModel(n, d, 2e-2, lambda_2, True, seed, None)

    # Training
    trainer = GolemTrainer(learning_rate)
    B_est = trainer.train(
        model, X, num_iter, interventional_data, checkpoint_iter, output_dir
    )

    B_est[np.abs(B_est) <= w_threshold] = 0

    if not equal_variances:
        # use found solution as init of
        logging.info("Find solution with Non equal variance formulation.")

        model = GolemModel(n, d, 2e-3, lambda_2, False, seed, B_est)

        # Training
        trainer = GolemTrainer(learning_rate)
        B_est = trainer.train(
            model, X, num_iter, interventional_data, checkpoint_iter, output_dir
        )

        B_est[np.abs(B_est) <= w_threshold] = 0

    return B_est
