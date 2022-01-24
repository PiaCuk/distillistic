import os
from copy import deepcopy

import torch
import torch.nn.functional as F
import wandb
from torch.cuda.amp import autocast
from tqdm import tqdm

from distillistic.utils import ClassifierMetrics, accuracy


class VirtualTeacher:
    """
    Implementation of the virtual teacher kowledge distillation framework from the paper
    "Revisit Knowledge Distillation: a Teacher-free Framework" https://arxiv.org/abs/1909.11723


    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
    :param correct_prob (float): Probability assigned to the correct class while generating soft labels for student training
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): DEPRECATED Directory for storing logs
    :param use_amp (bool): True to use Automated Mixed Precision
    """

    def __init__(
        self,
        student_model,
        train_loader,
        val_loader,
        optimizer_student,
        correct_prob=0.9,
        temp=10.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir=None,
        use_amp=False,
    ):

        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_student = optimizer_student
        self.correct_prob = correct_prob
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir
        self.amp = use_amp

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

        self.student_model = student_model.to(self.device)
        self.metrics = ClassifierMetrics(device=self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train_student(
        self,
        epochs=10,
        plot_losses=False,
        save_model=True,
        save_model_path="./experiments",
        use_scheduler=False,
        smooth_teacher=True,
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): DEPRECATED True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        :param use_scheduler (bool): True to use OneCycleLR during training
        :param smooth_teacher (bool): True to apply temperature smoothing and Softmax to virtual teacher
        """

        epoch_len = len(self.train_loader)
        log_freq = min(100, int(epoch_len / 10))

        self.student_model.train()
        if self.log:
            wandb.watch(self.student_model, log_freq=log_freq)

        best_acc = 0.0
        self.best_student_model_weights = deepcopy(
            self.student_model.state_dict())

        if use_scheduler:
            optim_lr = self.optimizer_student.param_groups[0]["lr"]
            scheduler_student = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer_student, max_lr=optim_lr, epochs=epochs, steps_per_epoch=epoch_len, pct_start=0.1)

        print("\nTraining student...")
        if plot_losses:
            print(
                "The argument plot_losses is deprecated. All metrics are logged to W&B.\n")

        for ep in tqdm(range(epochs), position=0):

            for batch_idx, (data, label) in enumerate(tqdm(self.train_loader, total=epoch_len, position=1)):

                data = data.to(self.device)
                label = label.to(self.device)

                self.optimizer_student.zero_grad(set_to_none=True)

                with autocast(enabled=self.amp):
                    student_out = self.student_model(data)
                    if isinstance(student_out, tuple):
                        student_out = student_out[0]

                loss = self.calculate_kd_loss(
                    student_out, label, smooth_teacher=smooth_teacher)
                if isinstance(loss, tuple):
                    loss, ce_loss, divergence = loss

                top1, top5, ece_loss, entropy, _ = self.metrics(student_out, label, topk=(1, 5))

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer_student)
                self.scaler.update()
                
                if use_scheduler:
                    scheduler_student.step()

                if self.log and batch_idx % log_freq == 0:
                    wandb.log({
                        "student/train_top1_acc": top1,
                        "student/train_top5_acc": top5,
                        "student/train_loss": loss,
                        "student/cross-entropy": ce_loss,
                        "student/divergence": divergence,
                        "student/calibration_error": ece_loss,
                        "student/entropy": entropy,
                        "student/lr": scheduler_student.get_last_lr()[0],
                        "student/distil_weight": self.distil_weight,
                        "epoch": ep,
                    })

                if batch_idx % int(2 * log_freq) == 0:
                    top1_val_acc, top5_val_acc = self.evaluate(verbose=False)

                    if top1_val_acc > best_acc:
                        best_acc = top1_val_acc.item()
                        self.best_student_model_weights = deepcopy(
                            self.student_model.state_dict()
                        )

                    if self.log:
                        wandb.log({
                            "student/val_top1_acc": top1_val_acc,
                            "student/val_top5_acc": top5_val_acc,
                            "student/best_acc": best_acc,
                            "epoch": ep,
                        })

        print(
            f"\nThe best student model validation accuracy {best_acc}")

        if save_model:
            torch.save(self.best_student_model_weights,
                       os.path.join(save_model_path, "student.pt"))

        return best_acc

    def calculate_kd_loss(self, y_pred_student, y_true, smooth_teacher=True):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_true (torch.FloatTensor): Original label
        """

        num_classes = y_pred_student.shape[1]

        virtual_teacher = torch.ones_like(y_pred_student, device=self.device)
        virtual_teacher = virtual_teacher * \
            (1 - self.correct_prob) / (num_classes - 1)
        for i in range(y_pred_student.shape[0]):
            virtual_teacher[i, y_true[i]] = self.correct_prob

        teacher_out = F.softmax(
            virtual_teacher / self.temp, dim=1) if smooth_teacher else virtual_teacher
        soft_student_out = F.log_softmax(y_pred_student / self.temp, dim=1)

        supervised = F.cross_entropy(y_pred_student, y_true)
        distillation = (self.temp ** 2) * F.kl_div(input=soft_student_out,
                                                   target=teacher_out,
                                                   reduction='batchmean', log_target=False)
        loss = (1 - self.distil_weight) * supervised + \
            self.distil_weight * distillation
        return loss, supervised, distillation

    def evaluate(self, verbose=True):
        """
        Evaluate method for printing accuracies of the trained network

        """

        model = deepcopy(self.student_model)
        model.eval()
        top1_acc = 0
        top5_acc = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                
                data = data.to(self.device)
                target = target.to(self.device)
                
                with autocast(enabled=self.amp):
                    output = model(data)
                
                if isinstance(output, tuple):
                    output = output[0]

                top1, top5 = accuracy(output, target, topk=(1, 5))
                top1_acc += top1
                top5_acc += top5

        top1_acc /= len(self.val_loader)
        top5_acc /= len(self.val_loader)

        if verbose:
            print("-" * 80)
            print(f"Accuracy: {top1_acc}")

        return top1_acc, top5_acc

    def get_parameters(self):
        """
        Get the number of parameters for the student network
        """

        student_params = sum(p.numel()
                             for p in self.student_model.parameters())

        print("-" * 80)
        print(
            f"Total parameters for the student network are: {student_params}")
