import argparse
from typing import Tuple

from . import sparsifiers
from . import structures
from . import equations
from . import utils

def parse_default_data_gen_args(
    argparser: argparse.ArgumentParser = None,
) -> Tuple[argparse.Namespace, argparse.ArgumentParser]:

    augmented_parser = (
        argparser if argparser else argparse.ArgumentParser(description="Data")
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

    parser.add_argument("--project", type=str, default="DAGuerreotype")
    parser.add_argument(
        "--nogpu",
        default=False,
        action="store_true",
        help="whether not to use gpu even if available",
    )
    parser.add_argument(
        "--wandb",
        default=False,
        action="store_true",
        help="whether logging in wandb console",
    )
    parser.add_argument("--entity", type=str, default="default") # Useful for WandB logging

    parser.add_argument(
        "--results_path",
        type=str,
        default="./results/",
        help="Path to save experimental results",
    )

    parser.add_argument("--model", default="daguerreo", choices=["daguerreo"])

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

    _add_from_module('--loss', utils)

    # -------------------------------------------------- MLP --------------------------------------------------

    parser.add_argument(
        "--hidden", type=int, default=10, help="Dimension of hidden layers"
    )

    # -------------------------------------------------- Training ---------------------------------------------
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        choices=["adam", "adamW", "sgd"],
    )
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--num_epochs", type=int, default=5000)

    parser.add_argument('--es_tol', type=int, default=100)
    parser.add_argument('--es_delta', type=float, default=1.e-4)

    parser.add_argument(
        "--pruning_reg", type=float, default=0.001, help="pruning penalty over edges"
    )
    parser.add_argument(  
        "--l2_theta",
        type=float,
        default=0.0005,
        help="l2 penalty for the structure vector",
    )
    parser.add_argument(  
        "--l2_eq",
        type=float,
        default=0.0005,
        help="l2 penalty over all models weights (not structure)",
    )

    # for bi-level optimization only
    parser.add_argument("--lr_theta", type=float, default=1e-1)
    parser.add_argument(
        "--eq_optimizer",
        default="sgd",
        type=str,
        choices=["adam", "adamW", "sgd"],
    )

    parser.add_argument('--es_tol_inner', type=int, default=10)
    parser.add_argument('--es_delta_inner', type=float, default=1.e-4)

    parser.add_argument("--num_inner_iters", type=int, default=200)

    # ------------------------------------------------ SparseMAP ------------------------------------------------

    parser.add_argument(
        "--init_theta", type=str, default="zeros", choices=["zeros", "variances"]
    )

    parser.add_argument(
        "--smap_init",
        action="store_true",
        default=False,
        help="SparseMap: whether draw first active solution randomly",
    )

    parser.add_argument(
        "--smap_iter_k",
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

    argparser.add_argument("--graph_types", type=list, default=["ER", "SF", "BP"], help="graph types to be tested")
    argparser.add_argument("--edge_ratios", type=list, default=[2, 4], help="number of expected edges per node to be tested")

    _, argparser = parse_default_data_gen_args(argparser=argparser)
    args, argparser = parse_default_model_args(argparser=argparser)

    if args.equations == "nonlinear":
        argparser.add_argument("--noise_models", type=list, default=[
            "mlp",
            "gp",
            "gp-add",
            "mim",
        ], help="noise models to be tested")
    else:
        argparser.add_argument("--noise_models", type=list, default=[
            "gauss",
            "gauss-heter",
            "exp",
            "gumbel",
            "uniform",
        ], help="noise models to be tested")

    return argparser
