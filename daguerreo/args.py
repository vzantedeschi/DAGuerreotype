import argparse
from typing import Tuple

from . import sparsifiers
from . import structures
from . import equations

# todo cleanup args


def parse_default_data_gen_args(
    argparser: argparse.ArgumentParser = None,
) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    # ------------------------------------------ Logistics --------------------------------------------
    augmented_parser = (
        argparser if argparser else argparse.ArgumentParser(description="GraphLearning")
    )

    augmented_parser.add_argument("--project", type=str, default="default")
    augmented_parser.add_argument("--entity", type=str, default="daguerro")
    augmented_parser.add_argument(
        "--gpu",
        default=True,
        action="store_true",
        help="whether to use gpu if available",
    )
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
            "--num_samples", type=int, default=1000, help="num of samples"
        )

        augmented_parser.add_argument(
            "--noise_scale",
            type=float,
            default=1,
            help="scale parameter of (independent) additive noise",
        )

        args, _ = augmented_parser.parse_known_args()

        augmented_parser.add_argument(
            "--s0", type=int, default=2 * args.num_nodes, help="expected num of edges"
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
        "--joint",
        default=False,
        action="store_true",
        help="whether optimizing ordering and graph jointly (alternated) or with bi-level formulation",
    )

    def _add_from_module(name, module, help_string=''):
        parser.add_argument(
            name,
            type=str,
            help=help_string,
            default=module.DEFAULT,
            choices=list(module.AVAILABLE.keys())
        )

    _add_from_module('--structure', structures)

    _add_from_module('--sparsifier', sparsifiers)

    _add_from_module('--equations', equations)

    # -------------------------------------------------- MLP --------------------------------------------------

    parser.add_argument(
        "--hidden", type=int, default=10, help="Dimensions of hidden layers"
    )

    # -------------------------------------------------- Training ---------------------------------------------
    # TODO check optimization hypers and how we use them in joint and bilevel
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        choices=["adam", "adamW", "sgd"],
    )
    parser.add_argument("--lr_theta", type=float, default=1e-1)
    parser.add_argument(
        "--eq_optimizer",
        default="sgd",
        type=str,
        choices=["adam", "adamW", "sgd"],
    )
    parser.add_argument("--lr", type=float, default=1e-1)

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="choice of device",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--pruning_reg", type=float, default=0.001, help="pruning penalty over graph"
    )
    parser.add_argument(  # use l2_reg instead
        "--l2_theta",
        type=float,
        default=0.0,
        help="l2 penalty for the structure vector",
    )
    parser.add_argument(  # use l2_reg instead
        "--l2_eq",
        type=float,
        default=0.0,
        help="l2 penalty over all models weights (not graph)",
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

    # ------------------------------------------------ SparseMAX ------------------------------------------------
    
    parser.add_argument('--smax_max_k', type=int, default=10)

    args, _ = parser.parse_known_args()
    return args, parser


def parse_pipeline_args() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(description="Benchmarking Pipeline")
    argparser.add_argument("--num_seeds", type=int, default=10)

    _, argparser = parse_default_data_gen_args(argparser=argparser)
    _, argparser = parse_default_model_args(argparser=argparser)

    return argparser


def parse_tuning_args() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(description="Hyper-Parameter Tuning")
    argparser.add_argument("--num_seeds", type=int, default=3,
                           help="number of seeds/datasets per trial (seed results are averaged)")
    argparser.add_argument("--num_trials", type=int, default=10, help="number of sets of hp to be tested")
    argparser.add_argument("--noise_models", type=list, default=[
                "gauss",
                "gauss-heter",
                "exp",
                "gumbel",
                "uniform",
            ], help="noise models to be tested")
    argparser.add_argument("--graph_types", type=list, default=["ER", "SF", "BP"], help="graph types to be tested")

    _, argparser = parse_default_data_gen_args(argparser=argparser)
    _, argparser = parse_default_model_args(argparser=argparser)

    return argparser
