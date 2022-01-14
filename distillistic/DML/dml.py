import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from distillistic.utils import ClassifierMetrics, accuracy


class DML:
    """
    Implementation of "Deep Mutual Learning" https://arxiv.org/abs/1706.00384

    :param student_cohort (list/tuple): Collection of student models
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param student_optimizers (list/tuple): Collection of Pytorch optimizers for training students
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): DEPRECATED Directory for storing logs
    :param use_ensemble (bool): If True, use ensemble target. Otherwise, compare students pairwise
    """

    def __init__(
        self,
        student_cohort,
        train_loader,
        val_loader,
        student_optimizers,
        loss_fn=nn.MSELoss(),
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir=None,
        use_ensemble=True,
    ):

        self.student_cohort = student_cohort
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.student_optimizers = student_optimizers
        self.loss_fn = loss_fn
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir
        self.use_ensemble = use_ensemble

        if self.use_ensemble:
            print("Using ensemble target for divergence loss.")

        if self.logdir is not None:
            print(
                "The argument logdir is deprecated. All metadata is stored in run folder.")

        if device.type == "cpu":
            self.device = torch.device("cpu")
            print("Device is set to CPU.")
        elif device.type == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Device is set to CUDA.")
        else:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = torch.device("cpu")

        for student in self.student_cohort:
            student.to(self.device)
        self.metrics = ClassifierMetrics().to(self.device)

    def ensemble_target(self, logits_list, j):
        # Calculate ensemble target given a list of logits, omitting the j'th element
        # num_logits = len(logits_list)
        # ensemble_target = torch.zeros(logits_list[j].shape).to(self.device)
        # for i, logits in enumerate(logits_list):
        #    if i != j:
        #        ensemble_target += (1 / (num_logits - 1)) * logits
        # return ensemble_target

        logits_list = logits_list[:j] + logits_list[j+1:]
        logits = torch.softmax(torch.stack(logits_list), dim=-1)
        return logits.mean(dim=0)

    def train_students(
        self,
        epochs=10,
        plot_losses=False,
        save_model=True,
        save_model_path="./experiments",
        use_scheduler=False,
        schedule_distil_weight=False,
    ):
        warm_up_pct = 0.1
        epoch_len = len(self.train_loader)
        log_freq = min(100, int(epoch_len / 10))

        for student_id, student in enumerate(self.student_cohort):
            student.train()
            if self.log:
                wandb.watch(student, log_freq=log_freq, idx=student_id)

        num_students = len(self.student_cohort)

        best_acc = 0.0
        self.best_student_model_weights = deepcopy(
            self.student_cohort[0].state_dict())
        self.best_student = self.student_cohort[0]
        self.best_student_id = 0

        if use_scheduler:
            self.student_schedulers = []

            for i in range(num_students):
                # Drop lr by 0.1 every 60 epochs (Zhang et al.)
                # self.student_schedulers.append(torch.optim.lr_scheduler.StepLR(
                #     self.student_optimizers[i], step_size=20, gamma=0.1))

                # Get learning rate from optimizer and create schedulers for each student
                optim_lr = self.student_optimizers[i].param_groups[0]["lr"]
                self.student_schedulers.append(torch.optim.lr_scheduler.OneCycleLR(
                    self.student_optimizers[i], max_lr=optim_lr, epochs=epochs, steps_per_epoch=epoch_len, pct_start=warm_up_pct))

        if schedule_distil_weight:
            self.target_distil_weight = self.distil_weight
            warm_up = int(2 * warm_up_pct * epochs)

        print("\nTraining students...")
        if plot_losses:
            print(
                "The argument plot_losses is deprecated. All metrics are logged to W&B.\n")

        cohort_step = 0

        for ep in tqdm(range(epochs), position=0):

            if schedule_distil_weight:
                if ep < warm_up:
                    self.distil_weight = (
                        (ep + 1e-8) / warm_up) * self.target_distil_weight
                else:
                    self.distil_weight = self.target_distil_weight

            for batch_idx, (data, label) in enumerate(tqdm(self.train_loader, total=epoch_len, position=1)):
            # for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                for optim in self.student_optimizers:
                    optim.zero_grad()

                # Forward passes to compute logits
                student_outputs = [n(data) for n in self.student_cohort]

                cohort_acc = 0

                for i in range(num_students):
                    student_loss = 0
                    if self.use_ensemble:
                        # Calculate ensemble target w/o applying softmax here
                        target = self.ensemble_target(student_outputs, i)
                        # Softmax should be applied in loss_fn
                        student_loss += self.loss_fn(
                            student_outputs[i], target.detach())
                    else:
                        # Calculate pairwise divergence
                        for j in range(num_students):
                            if i == j:
                                continue
                            else:
                                student_loss += (1 / (num_students - 1)) * self.loss_fn(
                                    student_outputs[i], student_outputs[j].detach())

                    ce_loss = F.cross_entropy(student_outputs[i], label)

                    top1, top5, ece_loss, entropy = self.metrics(student_outputs[i], label, topk=(1, 5))
                    cohort_acc += (1 / num_students) * top1

                    train_loss = (1 - self.distil_weight) * ce_loss + \
                        self.distil_weight * student_loss

                    train_loss.backward()
                    self.student_optimizers[i].step()
                    if use_scheduler:
                        self.student_schedulers[i].step()

                    if self.log and batch_idx % log_freq == 0:
                        wandb.log({
                            f"student{i}/train_top1_acc": top1,
                            f"student{i}/train_top5_acc": top5,
                            f"student{i}/train_loss": train_loss,
                            f"student{i}/cross-entropy": ce_loss,
                            f"student{i}/divergence": student_loss,
                            f"student{i}/calibration_error": ece_loss,
                            f"student{i}/entropy": entropy,
                            f"student{i}/lr": self.student_schedulers[i].get_last_lr()[0],
                            f"student{i}/distil_weight": self.distil_weight,
                            "epoch": ep,
                        }, step=cohort_step)

                if self.log:
                    wandb.log({
                        "cohort_train_acc": cohort_acc,
                        "epoch": ep,
                    }, step=cohort_step)
                cohort_step += 1

            cohort_val_acc = 0
            
            for student_id, student in enumerate(self.student_cohort):
                _, top1_val_acc, top5_val_acc = self._evaluate_model(student, verbose=False)
                cohort_val_acc += (1 / num_students) * top1_val_acc

                if top1_val_acc > best_acc:
                    best_acc = top1_val_acc.item()
                    best_student_id = student_id
                    self.best_student_model_weights = deepcopy(
                        student.state_dict())

                if self.log:
                    wandb.log({
                        f"student{student_id}/val_top1_acc": top1_val_acc,
                        f"student{student_id}/val_top5_acc": top5_val_acc,
                        "epoch": ep,
                    }, step=cohort_step)

            if self.log:
                wandb.log({
                    "best_student/val_acc": best_acc,
                    "cohort_val_acc": cohort_val_acc,
                    "epoch": ep,
                }, step=cohort_step)

        print(
            f"\nThe best student model is the model number {best_student_id} in the cohort. Validation accuracy {best_acc}")

        if save_model:
            if not os.path.isdir(save_model_path):
                os.mkdir(save_model_path)
            torch.save(self.best_student_model_weights, os.path.join(
                save_model_path, ("student" + str(best_student_id) + ".pt")))

        return best_acc

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        top1_acc = 0
        top5_acc = 0
        outputs = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                top1, top5 = accuracy(output, target, topk=(1, 5))
                top1_acc += top1
                top5_acc += top5

        top1_acc /= len(self.val_loader)
        top5_acc /= len(self.val_loader)

        if verbose:
            print(f"Accuracy: {top1_acc}")

        return outputs, top1_acc, top5_acc

    def evaluate(self, verbose=True):
        """
        Evaluate method for printing accuracies of the trained student networks

        """

        for i, student in enumerate(self.student_cohort):
            if verbose:
                print("-" * 80)
                print(f"Evaluating student {i}")

            model = deepcopy(student).to(self.device)
            out, top1, top5 = self._evaluate_model(model, verbose=verbose)

        return top1, top5

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """

        print("-" * 80)
        for i, student in enumerate(self.student_cohort):
            student_params = sum(p.numel() for p in student.parameters())
            print(
                f"Total parameters for the student network {i} are: {student_params}")
