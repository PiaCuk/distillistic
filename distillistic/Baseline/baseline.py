import os
from copy import deepcopy

import torch
import torch.nn as nn
import wandb
from torch.cuda.amp import autocast
from tqdm import tqdm

from distillistic.utils import ClassifierMetrics, accuracy

class Baseline:
    """
    Baseline for training a network on the target labels

    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer (torch.optim.*): Optimizer used for training student
    :param temp (float): Temperature parameter for distillation
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param use_amp (bool): True to use Automated Mixed Precision
    """
    def __init__(
        self,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        temp=1.0,
        device="cpu",
        log=False,
        use_amp=False,
    ) -> None:

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.temp = temp
        self.log = log
        self.amp = use_amp

        if self.log:
            self.log_freq = min(100, int(len(self.train_loader) / 10))

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

        self.ce_fn = nn.CrossEntropyLoss().to(self.device)
        self.metrics = ClassifierMetrics(device=self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
    
    def train_student(
        self,
        epochs=10,
        save_model=True,
        save_model_path="./experiments",
        use_scheduler=False
    ):
        """
        Function for training the baseline

        :param epochs (int): Number of epochs to train
        :param save_model (bool): True if you want to save the model
        :param save_model_path (str): Path where to store the model
        :param use_scheduler (bool): True to use OneCycleLR during training
        """

        self.student_model.train()
        if self.log:
            wandb.watch(self.student_model, log_freq=self.log_freq, idx=0)

        epoch_len = len(self.train_loader)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(
            self.student_model.state_dict())

        if use_scheduler:
            optim_lr = self.optimizer.param_groups[0]["lr"]
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=optim_lr, epochs=epochs, steps_per_epoch=epoch_len, pct_start=0.1)

        print("Training baseline... ")
        
        for ep in tqdm(range(epochs), position=0):

            for batch_idx, (data, label) in enumerate(tqdm(self.train_loader, total=epoch_len, position=1)):
                
                data = data.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.amp):
                    out = self.student_model(data)
                    if isinstance(out, tuple):
                        out = out[0]

                loss = self.ce_fn(out, label)

                top1, top5, ece_loss, entropy, virtual_kld = self.metrics(out, label, topk=(1, 5))

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if use_scheduler:
                    lr_scheduler.step()

                if self.log and batch_idx % self.log_freq == 0:
                    wandb.log({
                        "student/train_top1_acc": top1,
                        "student/train_top5_acc": top5,
                        "student/train_loss": loss,
                        "student/calibration_error": ece_loss,
                        "student/entropy": entropy,
                        "student/virtual_kld": virtual_kld,
                        "student/lr": lr_scheduler.get_last_lr()[0],
                        "epoch": ep,
                    })

                if batch_idx % int(2 * self.log_freq) == 0:
                    # Validation
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

        if save_model:
            torch.save(self.best_student_model_weights,
                       os.path.join(save_model_path, "student.pt"))

        return best_acc
    
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