import os
import statistics as s
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from distillistic.utils import ECELoss


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
    :param logdir (str): Directory for storing logs
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
        logdir="./Experiments",
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

        if self.log:
            self.writer = SummaryWriter(logdir)

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
        self.ece_loss = ECELoss(n_bins=15).to(self.device)

    def train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_path="./models/student.pt",
        use_scheduler=False,
        smooth_teacher=True,
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        :param use_scheduler (bool): True to use OneCycleLR during training
        :param smooth_teacher (bool): True to apply temperature smoothing and Softmax to virtual teacher
        """

        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(
            self.student_model.state_dict())

        if use_scheduler:
            optim_lr = self.optimizer_student.param_groups[0]["lr"]
            scheduler_student = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer_student, max_lr=optim_lr, epochs=epochs, steps_per_epoch=len(self.train_loader), pct_start=0.1)

        save_dir = os.path.dirname(save_model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("\nTraining student...")

        for ep in tqdm(range(epochs), position=0):
            epoch_loss = 0.0
            correct = 0
            student_ce_loss = []
            student_divergence = []
            student_entropy = []
            student_calibration = []

            epoch_len = int(length_of_dataset / self.train_loader.batch_size)

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                student_out = self.student_model(data)

                loss = self.calculate_kd_loss(
                    student_out, label, smooth_teacher=smooth_teacher)

                if isinstance(loss, tuple):
                    loss, ce_loss, divergence = loss
                    student_ce_loss.append(ce_loss.item())
                    student_divergence.append(divergence.item())

                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                student_calibration.append(
                    self.ece_loss(student_out, label).item())

                out_dist = Categorical(logits=student_out)
                entropy = out_dist.entropy().mean(dim=0)
                student_entropy.append(entropy.item())

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                if use_scheduler:
                    scheduler_student.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset

            epoch_val_acc = self.evaluate(verbose=False)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Loss/Train student", epoch_loss, ep)
                self.writer.add_scalar("Accuracy/Train student", epoch_acc, ep)
                self.writer.add_scalar("Accuracy/Validation student", epoch_val_acc, ep)
                self.writer.add_scalar("Loss/Cross-entropy student", s.mean(student_ce_loss), ep)
                self.writer.add_scalar("Loss/Divergence student", s.mean(student_divergence), ep)
                self.writer.add_scalar("Loss/Entropy student", s.mean(student_entropy), ep)
                self.writer.add_scalar("Loss/Calibration student", s.mean(student_calibration), ep)
                self.writer.add_scalar("Accuracy/Best student", best_acc, ep)
                if use_scheduler:
                    self.writer.add_scalar("Optimizer/lr student", scheduler_student.get_last_lr()[0], ep)

            loss_arr.append(epoch_loss)

        print(
            f"The best student model validation accuracy {best_acc}")

        if save_model:
            torch.save(self.best_student_model_weights,
                       os.path.join(save_model_path, "student.pt"))

        if plot_losses:
            plt.plot(loss_arr)

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
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset

        if verbose:
            print("-" * 80)
            print(f"Accuracy: {accuracy}")

        return accuracy

    def get_parameters(self):
        """
        Get the number of parameters for the student network
        """

        student_params = sum(p.numel()
                             for p in self.student_model.parameters())

        print("-" * 80)
        print(
            f"Total parameters for the student network are: {student_params}")
