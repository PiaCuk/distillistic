import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from distillistic import DML, VanillaKD, VirtualTeacher, resnet18, resnet50


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


def _create_optim(params, lr, adam=True):
    if adam:
        return torch.optim.AdamW(params, lr, betas=(0.9, 0.999))
    else:
        # Zhang et al. use no weight decay and nesterov=True
        return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=0.0001)


def create_distiller(algo, train_loader, test_loader, device, save_path, loss_fn=CustomKLDivLoss(), lr=0.01, distil_weight=0.5, temperature=10.0, num_students=2, pretrained=False, use_adam=True):
    """
    Create distillers for benchmarking.

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param test_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param device (str): Device used for training
    :param save_path (str): Directory for storing logs and saving models
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Not used for VanillaKD (BaseClass), as it is implemented internally
    :param lr (float): Learning rate
    :param distil_weight (float): Between 0 and 1
    :param temperature (float): temperature parameter for soft targets
    :param num_students (int): Number of students in cohort. Used for DML
    :param pretrained (bool): True to use pretrained torchvision models
    :param use_adam (bool): True to use Adam optim
    """
    resnet_params = {"num_classes": 1000, "pretrained": pretrained}

    if algo == "dml" or algo == "dml_e":
        # Define models
        student_cohort = [resnet18(**resnet_params) for i in range(num_students)]
        student_optimizers = [_create_optim(
            student_cohort[i].parameters(), lr, adam=use_adam) for i in range(num_students)]
        # Define DML with logging to Tensorboard
        distiller = DML(student_cohort, train_loader, test_loader, student_optimizers, loss_fn=loss_fn, distil_weight=distil_weight,
                        log=True, logdir=save_path, device=device, use_ensemble=True if algo == "dml_e" else False)
    elif algo == "tfkd":
        student = resnet18(**resnet_params)
        student_optimizer = _create_optim(
            student.parameters(), lr, adam=use_adam)
        # Define TfKD with logging to Tensorboard
        distiller = VirtualTeacher(student, train_loader, test_loader, student_optimizer,
            temp=temperature, distil_weight=distil_weight, log=True, logdir=save_path, device=device)
    else:
        teacher = resnet50(**resnet_params)
        student = resnet18(**resnet_params)

        teacher_optimizer = _create_optim(
            teacher.parameters(), lr, adam=use_adam)
        student_optimizer = _create_optim(
            student.parameters(), lr, adam=use_adam)
        # Define KD with logging to Tensorboard
        distiller = VanillaKD(teacher, student, train_loader, test_loader, teacher_optimizer, student_optimizer,
                              temp=temperature, distil_weight=distil_weight, log=True, logdir=save_path, device=device)

    return distiller
