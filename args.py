import argparse
from typing import Tuple


def parse_default_data_gen_args(
    argparser: argparse.ArgumentParser = None,
) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    # -------------------------------------------------- Logistics --------------------------------------------------
    augmented_parser = (
        argparser if argparser else argparse.ArgumentParser(description="GraphLearning")
    )

    augmented_parser.add_argument("--project", type=str, default="dag")
    augmented_parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="whether logging in wandb console",
    )

    augmented_parser.add_argument(
        "--results_path",
        type=str,
        default="./results/",
        help="Path to save experimental results",
    )

    augmented_parser.add_argument(
        "--dataset",
        type=str,
        default="sachs",
        choices=["synthetic", "sachs", "syntren"],
    )

    args, _ = augmented_parser.parse_known_args()

    if args.dataset == "synthetic":

        augmented_parser.add_argument(
            "--graph_type", type=str, default="SF", choices=["ER", "SF", "BP"]
        )
        augmented_parser.add_argument(
            "--sem_type",
            type=str,
            default="gauss",
            choices=[
                "mlp",
                "mim",
                "gp",
                "gp-add",
                "gauss",
                "gauss-heter",
                "exp",
                "gumbel",
                "uniform",
                "logistic",
                "poisson",
            ],
            help="SEM type to simulate samples from.",
        )
        augmented_parser.add_argument(
            "--num_nodes", type=int, default=10, help="num of nodes"
        )
        augmented_parser.add_argument(
            "--num_samples", type=int, default=200, help="num of samples"
        )

        augmented_parser.add_argument(
            "--noise_scale",
            type=float,
            default=0.5,
            help="scale parameter of (independent) additive noise",
        )

        args, _ = augmented_parser.parse_known_args()

        augmented_parser.add_argument(
            "--s0", type=int, default=2 * args.num_nodes, help="expected num of edges"
        )

        augmented_parser.add_argument(
            "--batch_size",
            type=int,
            default=args.num_samples,
            help="size of batches for stochastic optimization",
        )

    else:
        augmented_parser.add_argument(
            "--batch_size",
            type=int,
            default=1000,
            help="size of batches for stochastic optimization",
        )

    augmented_parser.add_argument(
        "--standardize",
        default=False,
        action="store_true",
        help="set to standardize data. default: only 0-center data",
    )

    args, _ = augmented_parser.parse_known_args()
    return args, augmented_parser


def parse_default_model_args(
    argparser: argparse.ArgumentParser = None,
) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = (
        argparser if argparser else argparse.ArgumentParser(description="Daguerreo")
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Daguerreo",
        choices=[
            "Daguerreo",
            "Oracle",
            "Random",
            "FixedAdj",
            "FixedPerm",
        ],
    )

    parser.add_argument(
        "--joint",
        default=False,
        action="store_true",
        help="whether optimizing ordering and graph jointly (alternated) or with bi-level formulation",
    )

    parser.add_argument(
        "--estimator",
        type=str,
        default="LARS",
        choices=[
            "LARS",
            "NN",
        ],
    )

    parser.add_argument(
        "--fixed_perm",
        type=str,
        default="variances",
        choices=["optimal", "variances"],  # TODO: add random ordering
    )

    # -------------------------------------------------- MLP --------------------------------------------------

    parser.add_argument(
        "--hidden", type=int, default=10, help="Dimensions of hidden layers"
    )

    parser.add_argument(
        "--nonlinear",
        default=False,
        action="store_true",
        help="whether use nonlinear graph",  # TODO what is this?
    )

    # -------------------------------------------------- Training --------------------------------------------------
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        choices=["adam", "adamW", "sgd"],
    )
    parser.add_argument("--lr_theta", type=float, default=1e-1)
    parser.add_argument("--lr", type=float, default=1e-3)

    args, _ = parser.parse_known_args()

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="choice of device",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--pruning_reg", type=float, default=0.01, help="pruning penalty over graph"
    )
    parser.add_argument(  # use l2_reg instead
        "--l2_reg",
        type=float,
        default=0.1,
        help="l2 penalty over all model weights (not graph)",
    )
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_inner_iters", type=int, default=100)

    argparser.add_argument(
        "--nev",
        action="store_true",
        default=False,
        help="compute likelihood assuming variable non-equal variance",
    )

    # ------------------------------------------------ SparseMAP ------------------------------------------------

    parser.add_argument(
        "--smap_tmp", type=float, default=1.0, help="SparseMap temperature"
    )

    parser.add_argument(
        "--smap_init_theta", type=str, default="zeros", choices=["zeros", "variances"]
    )

    parser.add_argument(
        "--smap_init",
        action="store_true",
        default=False,
        help="SparseMap: whether draw first active solution randomly",
    )

    parser.add_argument(
        "--smap_iter",
        type=int,
        default=100,
        help="SparseMap: number of iterations of QP solver",
    )

    args, _ = parser.parse_known_args()
    return args, parser


def parse_pipeline_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="Benchmarking Pipeline")
    argparser.add_argument("--num_seeds", type=int, default=10)

    _, argparser = parse_default_data_gen_args(argparser=argparser)
    _, argparser = parse_default_model_args(argparser=argparser)

    return argparser
