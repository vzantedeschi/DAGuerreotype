from typing import Iterable

from torch.optim import LBFGS, SGD, Adam, AdamW, Optimizer

def get_optimizer(params: Iterable, name: str, lr: float) -> Optimizer:
    
    str_to_optim = {
        "sgd": SGD,
        "adam": Adam,
        "adamW": AdamW
    }

    try:

        optim = str_to_optim[name]

    except Exception:

        raise NotImplementedError(
            f"The specified optimizer {name} is not implemented!"
        )

    return optim(params=params, lr=lr)

# --------------------------------------------------------- LOSSES

def nll_ev(output, target, dim=(-2, -1)):
    "negative log likelihood for Daguerro with equal variance"
    
    loss = (output - target).square()
    loss = loss.sum(dim=dim)

    result = torch.log(loss) * target.shape[-1] / 2
    return result