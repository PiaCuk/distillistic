import os
import statistics as s

import torch
import wandb

from distillistic.data import ImageNet_loader
from distillistic.distiller import create_distiller
from distillistic.utils import CustomKLDivLoss, set_seed


def ImageNet_experiment(
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
    classes=1000,
    use_amp=False,
    use_ffcv=False,
    downscale=1,
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
    :param use_weighted_dl (bool): TODO True to use weighted DataLoader with oversampling
    :param schedule_distil_weight (bool): True to increase distil_weight from 0 to distil_weight over warm-up period
    :param seed: Random seed
    :param classes (int): number of classes in training data. Default for ImageNet is 1000
    :param use_amp (bool): True to use Automated Mixed Precision
    :param use_ffcv (bool): True to load data with FFCV
    :param downscale (int): Downscaling factor. 1 for no downscaling
    """
    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set seed for all libraries and return torch.Generator
    g = set_seed(seed) if seed is not None else None
    workers = 12 if torch.cuda.is_available() else 4

    print(f"Creating DataLoaders. \nTraining with AMP is set to {use_amp}.")
    if downscale > 1:
        if algo != "vanilla":
            downscale_data = downscale
            downscale_model = 1
        else:
            downscale_data = 1
            downscale_model = downscale
    else:
        downscale_data = downscale
        downscale_model = downscale
    
    train_loader = ImageNet_loader(data_path, batch_size, device,
                                   train=True, generator=g, workers=workers, 
                                   use_amp=use_amp, use_ffcv=use_ffcv, downscale=downscale_data)
    test_loader = ImageNet_loader(data_path, batch_size, device,
                                  train=False, generator=g, workers=workers, 
                                  use_amp=use_amp, use_ffcv=use_ffcv, downscale=downscale_data)

    best_acc_list = []

    for i in range(runs):
        
        # TODO update naming to save models of multiple runs in one folder. 
        print(f"Starting run {i}")
        run_path = os.path.join(save_path, algo + str(i).zfill(3))
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        
        wandb.init(dir=run_path, config=locals(), project="distillistic", entity="piacuk", reinit=True)
        
        distiller = create_distiller(
            algo, train_loader, test_loader, device, num_classes=classes, loss_fn=loss_fn, lr=lr, 
            distil_weight=distil_weight, temperature=temperature, num_students=num_students, pretrained=use_pretrained,
            use_amp=use_amp, downscale=downscale_model
        )

        params = {"epochs": epochs, "save_model": True, "save_model_path": run_path, "use_scheduler": use_scheduler}

        if algo == "dml" or algo == "dml_e":
            # Run DML or DML_e
            acc = distiller.train_students(
                **params, schedule_distil_weight=schedule_distil_weight)
        elif algo == "tfkd":
            acc = distiller.train_student(**params, smooth_teacher=False)
        elif algo == "vanilla":
            if classes != 1000:
                distiller.train_teacher(epochs=int(epochs / 2), save_model_path=run_path, use_scheduler=use_scheduler)
            
            teacher1, teacher5 = distiller.evaluate(teacher=True)
            wandb.log({
                        "teacher/val_top1_acc": teacher1,
                        "teacher/val_top5_acc": teacher5,
                    }, commit=False)
            
            acc = distiller.train_student(**params)
        else:
            acc = distiller.train_student(**params)

        best_acc_list.append(acc)
        mean_acc = s.mean(best_acc_list)

        wandb.log({"Experiment mean acc": mean_acc})
        print(f"Mean validation accuracy of best model: {mean_acc}")
        
        return mean_acc
