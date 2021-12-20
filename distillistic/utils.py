import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class ECELoss(torch.nn.Module):
    """
    From https://github.com/gpleiss/temperature_scaling
    
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class CustomKLDivLoss(torch.nn.Module):
    def __init__(self, reduction='batchmean', log_target=False, apply_softmax=True) -> None:
        super(CustomKLDivLoss, self).__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.apply_softmax = apply_softmax

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.apply_softmax:
            target = torch.softmax(target, dim=-1)
        return F.kl_div(torch.log_softmax(input, dim=-1), target, reduction=self.reduction, log_target=self.log_target)


class SoftKLDivLoss(torch.nn.Module):
    def __init__(self, temp=20.0, reduction='batchmean', log_target=False) -> None:
        super(SoftKLDivLoss, self).__init__()
        self.temp = temp
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        soft_input = torch.log_softmax(input / self.temp, dim=-1)
        soft_target = torch.softmax(target / self.temp, dim=-1)
        # Multiply with squared temp so that KLD loss keeps proportion to CE loss
        return (self.temp ** 2) * F.kl_div(soft_input, soft_target, reduction=self.reduction, log_target=self.log_target)


def set_seed(seed=42, cuda_deterministic=False) -> torch.Generator:
    # See https://stackoverflow.com/a/64584503/8697610
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    # Generator used to control sampling of dataset
    # See https://pytorch.org/docs/stable/data.html#data-loading-randomness
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    return g
