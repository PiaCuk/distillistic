import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.categorical import Categorical


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
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin -
                                 accuracy_in_bin) * prop_in_bin

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


class ClassifierMetrics(torch.nn.Module):
    def __init__(self, device) -> None:
        super(ClassifierMetrics, self).__init__()
        self.device = device
        self.ece_loss = ECELoss(n_bins=15).to(self.device)
        self.virtual_prob = 0.9

    def virtual_teacher(self, pred: Tensor, target: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        virtual_teacher = torch.ones_like(pred, device=self.device)
        virtual_teacher = virtual_teacher * \
            (1 - self.virtual_prob) / (num_classes - 1)
        for i in range(pred.shape[0]):
            virtual_teacher[i, target[i]] = self.virtual_prob

        soft_student_out = F.log_softmax(pred, dim=1)

        kl_div = F.kl_div(input=soft_student_out, target=virtual_teacher,
                          reduction='batchmean', log_target=False)
        return kl_div

    def forward(self, pred: Tensor, target: Tensor, topk=(1,)) -> List[torch.FloatTensor]:
        """
        Returns a list with top k accuracy, ECE loss, entropy, and virtual teacher divergence of the predicted distribution.
        """
        metrics_list = accuracy(pred, target, topk) # List with top k accuracies
        metrics_list.append(self.ece_loss(pred, target)) # ECE loss

        out_dist = Categorical(logits=pred)
        metrics_list.append(out_dist.entropy().mean(dim=0)) # distribution entropy

        metrics_list.append(self.virtual_teacher(pred, target))

        return metrics_list


def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: prediction of the model, e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: ground truth
    :param topk: tuple of topk's to compute, e.g. (1, 2, 5) computes top 1, top 2 and top 5
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)
        y_pred = y_pred.t()

        target_reshaped = target.view(1, -1).expand_as(y_pred)
        correct = (y_pred == target_reshaped)
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        list_topk_accs = []
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
            topk_acc = tot_correct_topk / batch_size
            list_topk_accs.append(topk_acc)

        return list_topk_accs


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
