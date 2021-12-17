import os
import statistics as s

import torch

from distillistic.data import FMNIST_loader, FMNIST_weighted_loader
from distillistic.utils import CustomKLDivLoss, SoftKLDivLoss, create_distiller, set_seed


def distillation_experiment(
    algo,
    runs,
    epochs,
    batch_size,
    save_path,
    loss_fn=CustomKLDivLoss(),
    lr=0.005,
    distil_weight=0.5,
    temperature=10.0,
    num_students=2,
    use_pretrained=False,
    use_scheduler=False,
    use_weighted_dl=False,
    schedule_distil_weight=False,
    seed=None,
):
    """
    Universal main function for my Knowledge Distillation experiments

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param runs (int): Number of runs for each algorithm
    :param epochs (int): Number of epochs to train per run
    :param batch_size (int): Batch size for training
    :param save_path (str): Directory for storing logs and saving models
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Only used for DML
    :param lr (float): Learning rate
    :param distil_weight (float): Weight for distillation loss. Cross-entropy will be weighted with (1 - distil_weight)
    :param temperature (float): temperature parameter for soft targets
    :param num_students (int): Number of students in cohort. Used for DML
    :param use_pretrained (bool): Use pretrained teacher for VanillaKD
    :param use_scheduler (bool): True to decrease learning rate during training
    :param use_weighted_dl (bool): True to use weighted DataLoader with oversampling
    :param schedule_distil_weight (bool): True to increase distil_weight from 0 to distil_weight over warm-up period
    :param seed: Random seed
    """
    # Set seed for all libraries and return torch.Generator
    g = set_seed(seed) if seed is not None else None

    # Create DataLoaders
    if use_weighted_dl:
        train_loader = FMNIST_weighted_loader(
            batch_size, train=True, generator=g, workers=15)
        test_loader = FMNIST_weighted_loader(
            batch_size, train=False, generator=g, workers=15)
    else:
        train_loader = FMNIST_loader(
            batch_size, train=True, generator=g, workers=15)
        test_loader = FMNIST_loader(
            batch_size, train=False, generator=g, workers=15)

    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_acc_list = []

    for i in range(runs):
        print(f"Starting run {i}")
        run_path = os.path.join(save_path, algo + str(i).zfill(3))

        distiller = create_distiller(
            algo, train_loader, test_loader, device, save_path=run_path, loss_fn=loss_fn, lr=lr,
            distil_weight=distil_weight, temperature=temperature, num_students=num_students, pretrained=use_pretrained)

        params = {"epochs": epochs, "plot_losses": False, "save_model": True,
                "save_model_path": run_path, "use_scheduler": use_scheduler}

        if algo == "dml" or algo == "dml_e":
            # Run DML or DML_e
            acc = distiller.train_students(**params, schedule_distil_weight=schedule_distil_weight)
        elif algo == "tfkd":
            acc = distiller.train_student(**params, smooth_teacher=False)
        else:
            if not use_pretrained:
                distiller.train_teacher(**params)
            acc = distiller.train_student(**params)

        best_acc_list.append(acc)
        mean_acc = s.mean(best_acc_list)

        print(f"Mean validation accuracy of best model: {mean_acc}")
        return mean_acc
