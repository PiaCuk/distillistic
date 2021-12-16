import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt
from copy import deepcopy
import os
from tqdm import tqdm
import statistics as s

from KD_Lib.KD.common.utils import ECELoss


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
    :param logdir (str): Directory for storing logs
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
        device=torch.device("cpu"),
        log=False,
        logdir="./Experiments",
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
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
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_path="./models/teacher.pt",
        use_scheduler=False
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_path (str): Path where you want to store the teacher model
        :param use_scheduler (bool): True to use OneCycleLR during training
        """
        self.teacher_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        if use_scheduler:
            optim_lr = self.optimizer_teacher.param_groups[0]["lr"]
            scheduler_teacher = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer_teacher, max_lr=optim_lr, epochs=epochs, steps_per_epoch=len(self.train_loader), pct_start=0.1)

        save_dir = os.path.dirname(save_model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Teacher... ")

        for ep in tqdm(range(epochs), position=0):
            epoch_loss = 0.0
            epoch_calibration = 0.0
            correct = 0
            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                out = self.teacher_model(data)

                if isinstance(out, tuple):
                    out = out[0]

                epoch_calibration += (1 / length_of_dataset) * self.ece_loss(out, label).item()

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                loss = self.ce_fn(out, label)

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                if use_scheduler:
                    scheduler_teacher.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset

            epoch_val_acc = self.evaluate(teacher=True, verbose=False)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Loss/Train teacher", epoch_loss, ep)
                self.writer.add_scalar("Loss/Calibration teacher", epoch_calibration, ep)
                self.writer.add_scalar("Accuracy/Train teacher", epoch_acc, ep)
                self.writer.add_scalar("Accuracy/Validation teacher", epoch_val_acc, ep)
                if use_scheduler:
                    self.writer.add_scalar("Optimizer/lr teacher", scheduler_teacher.get_last_lr()[0], ep)

            loss_arr.append(epoch_loss)

            self.post_epoch_call(ep)

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        if save_model:
            torch.save(self.teacher_model.state_dict(), os.path.join(save_model_path, "teacher.pt"))
        if plot_losses:
            plt.plot(loss_arr)

    def _train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_path="./models/student.pt",
        use_scheduler=False
    ):
        """
        Function to train student model - for internal use only.

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_path (str): Path where you want to save the student model
        :param use_scheduler (bool): True to use OneCycleLR during training
        """
        self.teacher_model.eval()
        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        if use_scheduler:
            optim_lr = self.optimizer_student.param_groups[0]["lr"]
            scheduler_student = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer_student, max_lr=optim_lr, epochs=epochs, steps_per_epoch=len(self.train_loader), pct_start=0.1)

        save_dir = os.path.dirname(save_model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")

        for ep in tqdm(range(epochs), position=0):
            epoch_loss = 0.0
            correct = 0
            student_ce_loss = []
            student_divergence = []
            student_entropy = []
            student_calibration = []

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                student_out = self.student_model(data)
                teacher_out = self.teacher_model(data)

                loss = self.calculate_kd_loss(student_out, teacher_out, label)
                
                if isinstance(loss, tuple):
                    loss, ce_loss, divergence = loss
                    student_ce_loss.append(ce_loss.item())
                    student_divergence.append(divergence.item())
                
                out_dist = Categorical(logits=student_out)
                entropy = out_dist.entropy().mean(dim=0)
                student_entropy.append(entropy.item())

                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                student_calibration.append(self.ece_loss(student_out, label).item())
                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                if use_scheduler:
                    scheduler_student.step()

                epoch_loss += loss.item()

            epoch_acc = correct / length_of_dataset

            epoch_val_acc = self.evaluate(teacher=False, verbose=False)

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
                if use_scheduler:
                    self.writer.add_scalar("Optimizer/lr student", scheduler_student.get_last_lr()[0], ep)

            loss_arr.append(epoch_loss)

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), os.path.join(save_model_path, "student.pt"))
        if plot_losses:
            plt.plot(loss_arr)

    def train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_path="./models/student.pt",
        use_scheduler=False
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_path (str): Path where you want to save the student model
        :param use_scheduler (bool): True to use OneCycleLR during training
        """
        self._train_student(epochs, plot_losses, save_model, save_model_path, use_scheduler)

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
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

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
