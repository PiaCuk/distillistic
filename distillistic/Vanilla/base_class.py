import os
from copy import deepcopy

import torch
import torch.nn as nn
import wandb
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from distillistic.utils import ECELoss


class BaseClass:
    """
    Basic implementation of a general Knowledge Distillation framework

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): DEPRECATED Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        loss_fn=nn.KLDivLoss(),
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir=None,
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir

        if self.logdir is not None:
            print(
                "The argument logdir is deprecated. All metadata is stored in run folder.")
        if self.log:
            self.log_freq = 100

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

        if teacher_model:
            self.teacher_model = teacher_model.to(self.device)
        else:
            print("Warning!!! Teacher is NONE.")

        self.student_model = student_model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.ce_fn = nn.CrossEntropyLoss().to(self.device)
        self.ece_loss = ECELoss(n_bins=15).to(self.device)

    def train_teacher(
        self,
        epochs=10,
        plot_losses=False,
        save_model=True,
        save_model_path="./experiments",
        use_scheduler=False
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): DEPRECATED True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_path (str): Path where you want to store the teacher model
        :param use_scheduler (bool): True to use OneCycleLR during training
        """

        self.teacher_model.train()
        if self.log:
            wandb.watch(self.teacher_model, log_freq=self.log_freq, idx=0)

        epoch_len = len(self.train_loader)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(
            self.teacher_model.state_dict())

        if use_scheduler:
            optim_lr = self.optimizer_teacher.param_groups[0]["lr"]
            scheduler_teacher = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer_teacher, max_lr=optim_lr, epochs=epochs, steps_per_epoch=epoch_len, pct_start=0.1)

        print("Training Teacher... ")
        if plot_losses:
            print(
                "The argument plot_losses is deprecated. All metrics are logged to W&B.\n")

        for ep in tqdm(range(epochs), position=0):

            for batch_idx, (data, label) in enumerate(tqdm(self.train_loader, total=epoch_len, position=1)):
                data = data.to(self.device)
                label = label.to(self.device)

                out = self.teacher_model(data)
                if isinstance(out, tuple):
                    out = out[0]

                loss = self.ce_fn(out, label)

                ece_loss = self.ece_loss(out, label).item()

                out_dist = Categorical(logits=out)
                entropy = out_dist.entropy().mean(dim=0)

                preds = out.argmax(dim=1, keepdim=True)
                train_acc = preds.eq(label.view_as(
                    preds)).sum().item() / len(preds)

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()
                if use_scheduler:
                    scheduler_teacher.step()

                if self.log and batch_idx % self.log_freq == 0:
                    wandb.log({
                        "teacher/train_acc": train_acc,
                        "teacher/train_loss": loss,
                        "teacher/calibration_error": ece_loss,
                        "teacher/entropy": entropy,
                        "teacher/lr": scheduler_teacher.get_last_lr()[0],
                        "teacher/distil_weight": self.distil_weight,
                        "epoch": ep,
                    })

            epoch_val_acc = self.evaluate(teacher=True, verbose=False)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            if self.log:
                wandb.log({
                    "teacher/val_acc": epoch_val_acc,
                    "teacher/best_acc": best_acc,
                    "epoch": ep,
                })

            self.post_epoch_call(ep)

        if save_model:
            torch.save(self.best_teacher_model_weights,
                       os.path.join(save_model_path, "teacher.pt"))

        return best_acc

    def _train_student(
        self,
        epochs=10,
        save_model=True,
        save_model_path="./experiments",
        use_scheduler=False
    ):
        """
        Function to train student model - for internal use only.

        :param epochs (int): Number of epochs you want to train the teacher
        :param save_model (bool): True if you want to save the student model
        :param save_model_path (str): Path where you want to save the student model
        :param use_scheduler (bool): True to use OneCycleLR during training
        """
        self.teacher_model.eval()
        self.student_model.train()
        if self.log:
            wandb.watch(self.student_model, log_freq=self.log_freq, idx=1)

        epoch_len = len(self.train_loader)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(
            self.student_model.state_dict())

        if use_scheduler:
            optim_lr = self.optimizer_student.param_groups[0]["lr"]
            scheduler_student = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer_student, max_lr=optim_lr, epochs=epochs, steps_per_epoch=epoch_len, pct_start=0.1)

        print("Training Student...")

        for ep in tqdm(range(epochs), position=0):

            for batch_idx, (data, label) in enumerate(tqdm(self.train_loader, total=epoch_len, position=1)):

                data = data.to(self.device)
                label = label.to(self.device)

                student_out = self.student_model(data)
                teacher_out = self.teacher_model(data)
                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                loss = self.calculate_kd_loss(student_out, teacher_out, label)
                if isinstance(loss, tuple):
                    loss, ce_loss, divergence = loss

                ece_loss = self.ece_loss(student_out, label).item()

                out_dist = Categorical(logits=student_out)
                entropy = out_dist.entropy().mean(dim=0)

                preds = student_out.argmax(dim=1, keepdim=True)
                train_acc = preds.eq(label.view_as(
                    preds)).sum().item() / len(preds)

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()
                if use_scheduler:
                    scheduler_student.step()

                if self.log and batch_idx % self.log_freq == 0:
                    wandb.log({
                        "student/train_acc": train_acc,
                        "student/train_loss": loss,
                        "student/cross-entropy": ce_loss,
                        "student/divergence": divergence,
                        "student/calibration_error": ece_loss,
                        "student/entropy": entropy,
                        "student/lr": scheduler_student.get_last_lr()[0],
                        "student/distil_weight": self.distil_weight,
                        "epoch": ep,
                    })

            epoch_val_acc = self.evaluate(teacher=False, verbose=False)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                wandb.log({
                    "student/val_acc": epoch_val_acc,
                    "student/best_acc": best_acc,
                    "epoch": ep,
                })

        print(
            f"The best student model validation accuracy {best_acc}")

        if save_model:
            torch.save(self.best_student_model_weights,
                       os.path.join(save_model_path, "student.pt"))

        return best_acc

    def train_student(
        self,
        epochs=10,
        plot_losses=False,
        save_model=True,
        save_model_path="./experiments",
        use_scheduler=False
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): DEPRECATED True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_path (str): Path where you want to save the student model
        :param use_scheduler (bool): True to use OneCycleLR during training
        """

        if plot_losses:
            print(
                "The argument plot_losses is deprecated. All metrics are logged to W&B.\n")

        self._train_student(epochs, save_model, save_model_path, use_scheduler)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset

        if verbose:
            print("-" * 80)
            print("Validation Accuracy: {}".format(accuracy))

        return outputs, accuracy

    def evaluate(self, teacher=False, verbose=True):
        """
        Evaluate method for printing accuracies of the trained network

        :param teacher (bool): True if you want accuracy of the teacher network
        """
        if teacher:
            model = deepcopy(self.teacher_model).to(self.device)
        else:
            model = deepcopy(self.student_model).to(self.device)
        _, accuracy = self._evaluate_model(model, verbose=verbose)

        return accuracy

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel()
                             for p in self.teacher_model.parameters())
        student_params = sum(p.numel()
                             for p in self.student_model.parameters())

        print("-" * 80)
        print("Total parameters for the teacher network are: {}".format(teacher_params))
        print("Total parameters for the student network are: {}".format(student_params))

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.

        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass
