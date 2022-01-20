import glob
import os
import statistics as s

import torch
import wandb

from distillistic.data import FMNIST_loader
from distillistic.distiller import create_distiller
from distillistic.utils import CustomKLDivLoss, set_seed


def FMNIST_experiment(
    algo,
    runs,
    epochs,
    batch_size,
    data_path,
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
    classes=10,
):
    """
    Universal main function for my Knowledge Distillation experiments

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", "tfkd", else VanillaKD
    :param runs (int): Number of runs for each algorithm
    :param epochs (int): Number of epochs to train per run
    :param batch_size (int): Batch size for training
    :param data_path (str): Directory from which to load the data
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
    :param classes (int): number of classes in training data. Default for ImageNet is 1000
    """
    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set seed for all libraries and return torch.Generator
    g = set_seed(seed) if seed is not None else None
    workers = 12 if torch.cuda.is_available() else 4

    # Create DataLoaders
    train_loader = FMNIST_loader(
        data_path, batch_size, train=True, generator=g, 
        workers=workers, weighted_sampler=use_weighted_dl
    )
    test_loader = FMNIST_loader(
        data_path, batch_size, train=False, generator=g, 
        workers=workers, weighted_sampler=use_weighted_dl
    )

    best_acc_list = []

    for i in range(runs):
        print(f"Starting run {i}")
        run_path = os.path.join(save_path, algo + str(i).zfill(3))
        if not os.path.isdir(run_path):
            os.makedirs(run_path)

        wandb.init(dir=run_path, config=locals(), project="Fashion-MNIST", entity="piacuk", reinit=True)

        distiller = create_distiller(
            algo, train_loader, test_loader, device, num_classes=classes, loss_fn=loss_fn, lr=lr, 
            distil_weight=distil_weight, temperature=temperature, num_students=num_students, pretrained=use_pretrained
        )

        params = {"epochs": epochs, "plot_losses": False, "save_model": True,
                  "save_model_path": run_path, "use_scheduler": use_scheduler}

        if algo == "dml" or algo == "dml_e":
            # Run DML or DML_e
            acc = distiller.train_students(
                **params, schedule_distil_weight=schedule_distil_weight)
        elif algo == "tfkd":
            acc = distiller.train_student(**params, smooth_teacher=False)
        else:
            state_dict = torch.load(
                "./experiments/Fashion-MNIST/pretrained_models/vanilla000/teacher.pt"
            )
            distiller.teacher_model.load_state_dict(state_dict)
            # distiller.train_teacher(epochs=int(epochs / 2), save_model_path=run_path, use_scheduler=use_scheduler)
            
            teacher1, teacher5 = distiller.evaluate(teacher=True)
            wandb.log({
                        "teacher/val_top1_acc": teacher1,
                        "teacher/val_top5_acc": teacher5,
                    }, commit=False)
            
            acc = distiller.train_student(**params)

        best_acc_list.append(acc)
        mean_acc = s.mean(best_acc_list)

        wandb.log({"Experiment mean acc": mean_acc})
        print(f"Mean validation accuracy of best model: {mean_acc}")
        
        return mean_acc


def FMNIST_test(
    algo,
    load_dir,
    batch_size,
    data_path,
    loss_fn=CustomKLDivLoss(),
    lr=0.005,
    distil_weight=0.5,
    temperature=10.0,
    use_weighted_dl=False,
    seed=None,
):
    """
    Evaluation function for trained and saved models

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param load_dir (str):
    :param batch_size (int): Batch size for training
    :param data_path (str): Directory from which to load the data
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Not used for VanillaKD (BaseClass), as it is implemented internally
    :param lr (float): Learning rate
    :param distil_weight (float): Between 0 and 1
    :param temperature (float): temperature parameter for soft targets
    :param use_weighted_dl (bool): True to use weighted DataLoader with oversampling
    :param seed:    
    """
    # Set seed for all libraries and return torch.Generator
    g = set_seed(seed) if seed is not None else None
    workers = 12 if torch.cuda.is_available() else 4

    # Create DataLoaders
    train_loader = FMNIST_loader(
        data_path, batch_size, train=True, generator=g, 
        workers=workers, weighted_sampler=use_weighted_dl
    )
    test_loader = FMNIST_loader(
        data_path, batch_size, train=False, generator=g, 
        workers=workers, weighted_sampler=use_weighted_dl
    )

    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    distiller = create_distiller(
        algo, train_loader, test_loader, device, save_path=load_dir, num_classes=10,
        loss_fn=loss_fn, lr=lr, distil_weight=distil_weight, temperature=temperature, num_students=1)

    if algo == "vanilla":
        state_dict = torch.load(os.path.join(
            load_dir, "teacher.pt"), map_location=device)
        distiller.teacher_model.load_state_dict(state_dict)
        state_dict = torch.load(os.path.join(
            load_dir, "student.pt"), map_location=device)
        distiller.student_model.load_state_dict(state_dict)

        print("Evaluating teacher... \n")
        distiller.evaluate(teacher=True)
        print("Evaluating student... \n")
        distiller.evaluate(teacher=False)
    elif algo == "tfkd":
        state_dict = torch.load(os.path.join(
            load_dir, "student.pt"), map_location=device)
        distiller.student_model.load_state_dict(state_dict)
        distiller.evaluate()
    else:
        model_pt = glob.glob(os.path.join(load_dir, "student*.pt"))[0]
        state_dict = torch.load(model_pt, map_location=device)
        distiller.student_cohort[0].load_state_dict(state_dict)
        distiller.evaluate()
