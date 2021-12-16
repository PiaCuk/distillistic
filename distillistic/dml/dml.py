import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from distillistic import ECELoss


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
    :param logdir (str): Directory for storing logs
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
        logdir="./Experiments",
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

        for student in self.student_cohort:
            student.to(self.device)
        self.ece_loss = ECELoss(n_bins=15).to(self.device)

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
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_path="./Experiments",
        use_scheduler=False,
        schedule_distil_weight=False,
    ):
        for student in self.student_cohort:
            student.train()

        loss_arr = []

        num_students = len(self.student_cohort)
        length_of_dataset = len(self.train_loader.dataset)
        epoch_len = len(self.train_loader) # int(length_of_dataset / self.train_loader.batch_size)

        best_acc = 0.0
        self.best_student_model_weights = deepcopy(
            self.student_cohort[0].state_dict())
        self.best_student = self.student_cohort[0]
        self.best_student_id = 0

        warm_up_pct = 0.1
                
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

        for ep in tqdm(range(epochs), position=0):
            epoch_loss = 0.0
            correct = 0
            cohort_ce_loss = [0 for s in range(num_students)]
            cohort_divergence = [0 for s in range(num_students)]
            cohort_entropy = [0 for s in range(num_students)]
            cohort_calibration = [0 for s in range(num_students)]

            if schedule_distil_weight:
                if ep < warm_up:
                    self.distil_weight = ((ep + 1e-8) / warm_up) * self.target_distil_weight
                else:
                    self.distil_weight = self.target_distil_weight

            #for (data, label) in tqdm(self.train_loader, total=epoch_len, position=1):
            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                for optim in self.student_optimizers:
                    optim.zero_grad()

                # Forward passes to compute logits
                student_outputs = [n(data) for n in self.student_cohort]

                avg_student_loss = 0

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

                    # Running average of both loss summands
                    cohort_ce_loss[i] += (1 / epoch_len) * ce_loss
                    cohort_divergence[i] += (1 / epoch_len) * student_loss

                    cohort_calibration[i] += (1 / epoch_len) * \
                        self.ece_loss(student_outputs[i], label).item()

                    # Running average of output entropy
                    output_distribution = Categorical(
                        logits=student_outputs[i])
                    entropy = output_distribution.entropy().mean(dim=0)
                    cohort_entropy[i] += (1 / epoch_len) * entropy

                    student_loss = (1 - self.distil_weight) * ce_loss + \
                        self.distil_weight * student_loss
                    avg_student_loss += (1 / num_students) * student_loss

                    student_loss.backward()
                    self.student_optimizers[i].step()
                    if use_scheduler:
                        self.student_schedulers[i].step()

                predictions = []
                correct_preds = []
                for i in range(num_students):
                    predictions.append(
                        student_outputs[i].argmax(dim=1, keepdim=True))
                    correct_preds.append(
                        predictions[i].eq(label.view_as(
                            predictions[i])).sum().item()
                    )

                correct += sum(correct_preds) / len(correct_preds)

                epoch_loss += avg_student_loss

            epoch_acc = correct / length_of_dataset
            # TODO log training accuracy for each student separately

            for student_id, student in enumerate(self.student_cohort):
                _, epoch_val_acc = self._evaluate_model(student, verbose=False)

                if epoch_val_acc > best_acc:
                    best_acc = epoch_val_acc
                    self.best_student_model_weights = deepcopy(
                        student.state_dict())
                    self.best_student = student
                    self.best_student_id = student_id

                if self.log:
                    self.writer.add_scalar(
                        "Accuracy/Validation student"+str(student_id), epoch_val_acc, ep)
                    self.writer.add_scalar(
                        "Loss/Cross-entropy student"+str(student_id), cohort_ce_loss[student_id], ep)
                    self.writer.add_scalar(
                        "Loss/Divergence student"+str(student_id), cohort_divergence[student_id], ep)
                    self.writer.add_scalar(
                        "Loss/Entropy student"+str(student_id), cohort_entropy[student_id], ep)
                    self.writer.add_scalar(
                        "Loss/Calibration student"+str(student_id), cohort_calibration[student_id], ep)
                    
                    if use_scheduler:
                        self.writer.add_scalar(
                            "Optimizer/lr student"+str(student_id), self.student_schedulers[student_id].get_last_lr()[0], ep)

            if self.log:
                self.writer.add_scalar("Loss/Train average", epoch_loss, ep)
                self.writer.add_scalar("Accuracy/Train average", epoch_acc, ep)
                self.writer.add_scalar("Optimizer/Distillation weight", self.distil_weight, ep)
                self.writer.add_scalar("Accuracy/Best student", self._evaluate_model(self.best_student, verbose=False)[1], ep)

            loss_arr.append(epoch_loss)

        self.best_student.load_state_dict(self.best_student_model_weights)
        if save_model:
            print(
                f"The best student model is the model number {self.best_student_id} in the cohort")
            if not os.path.isdir(save_model_path):
                os.mkdir(save_model_path)
            torch.save(self.best_student.state_dict(), os.path.join(
                save_model_path, ("student" + str(self.best_student_id) + ".pt")))
        if plot_losses:
            plt.plot(loss_arr)

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
            print(f"Accuracy: {accuracy}")

        return outputs, accuracy

    def evaluate(self):
        """
        Evaluate method for printing accuracies of the trained student networks

        """

        for i, student in enumerate(self.student_cohort):
            print("-" * 80)
            model = deepcopy(student).to(self.device)
            print(f"Evaluating student {i}")
            out, acc = self._evaluate_model(model)

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """

        print("-" * 80)
        for i, student in enumerate(self.student_cohort):
            student_params = sum(p.numel() for p in student.parameters())
            print(
                f"Total parameters for the student network {i} are: {student_params}")
