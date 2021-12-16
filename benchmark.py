import os

import torch

from main import main
from utils import (CustomKLDivLoss, SoftKLDivLoss, create_dataloader,
                   create_distiller, set_seed)

# TODO implement dynamic temperature in TfKD


def benchmark(algo, runs, epochs, batch_size, save_path, loss_fn=CustomKLDivLoss(), num_students=2, use_adam=True, seed=None, use_scheduler=False):
    """
    Main function to call for benchmarking.
    !!use main.py!!

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param runs (int): Number of runs for each algorithm
    :param epochs (int): Number of epochs to train per run
    :param batch_size (int): Batch size for training
    :param save_path (str): Directory for storing logs and saving models
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Not used for VanillaKD (BaseClass), as it is implemented internally
    :param num_students (int): Number of students in cohort. Used for DML
    :param use_adam (bool): True to use Adam optim
    :param seed:
    :param use_scheduler (bool): True to decrease learning rate during training
    """
    if seed is not None:
        g = set_seed(seed)

    train_loader = create_dataloader(
        batch_size, train=True, generator=g if seed is not None else None)

    test_loader = create_dataloader(
        batch_size, train=False, generator=g if seed is not None else None)

    # Set device to be trained on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(runs):
        print(f"Starting run {i}")
        run_path = os.path.join(save_path, algo + str(i).zfill(3))
        distiller = create_distiller(
            algo, train_loader, test_loader, device, save_path=run_path, loss_fn=loss_fn, lr=0.005, distil_weight=0.5, num_students=num_students, use_adam=use_adam)

        # epochs, plot_losses, save_model, save_model_path, use_scheduler
        param_list = [epochs, False, True, run_path, use_scheduler]

        if algo == "dml" or algo == "dml_e":
            # Run DML or DML_e
            distiller.train_students(*param_list)
        elif algo == "tfkd":
            distiller.train_student(*param_list)
        else:
            distiller.train_teacher(*param_list)

            print("Train student with temperature " + str(distiller.temp))
            distiller.train_student(*param_list)


"""
Note that benchmark was previously main
Now using universal main function for benchmarking

    # First round of experiments
    main("dml", 5, 100, 1024, "/data1/9cuk/kd_lib/session6",
            loss_fn=SoftKLDivLoss(temp=20), num_students=3)
    main("dml_e", 5, 100, 1024, "/data1/9cuk/kd_lib/session6",
            loss_fn=SoftKLDivLoss(temp=20), num_students=3)
    main("vanilla", 5, 100, 1024, "/data1/9cuk/kd_lib/session3_3")
    main("tfkd", 5, 100, 1024, "/data1/9cuk/kd_lib/session7")

    # First calibration experiments
    main("dml", 5, 100, 1024, "/data1/9cuk/kd_lib/calibration1",
        loss_fn=CustomKLDivLoss(), num_students=3, seed=42)
    main("dml_e", 5, 100, 1024, "/data1/9cuk/kd_lib/calibration1",
        loss_fn=CustomKLDivLoss(), num_students=3, seed=42)
    main("vanilla", 5, 100, 1024, "/data1/9cuk/kd_lib/calibration1", seed=42)
    main("tfkd", 5, 10, 1024, "/data1/9cuk/kd_lib/calibration0", seed=42)

    # Super-convergence with OneCycleLR
    main("dml", 5, 100, 1024, "/data1/9cuk/kd_lib/super_convergence0",
        loss_fn=CustomKLDivLoss(), num_students=3, seed=42, use_scheduler=True)

    # Get all results for thesis
    main("dml", 5, 100, 1024, "/data1/9cuk/kd_lib/super_convergence1",
        loss_fn=CustomKLDivLoss(), num_students=3, seed=42, use_scheduler=True)
    main("vanilla", 5, 100, 1024, "/data1/9cuk/kd_lib/calibration3",
        loss_fn=CustomKLDivLoss(), num_students=3, seed=42,  use_scheduler=False)
    main("tfkd", 5, 100, 1024, "/data1/9cuk/kd_lib/super_convergence1",
        loss_fn=CustomKLDivLoss(), num_students=3, seed=42, use_scheduler=True)
"""
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Use new universal main
    main(
        "dml",
        5,
        100,
        1024,
        "/data1/9cuk/kd_lib/hyperparams/adamw-003-10-False",
        loss_fn=CustomKLDivLoss(apply_softmax=True), # only used for DML
        lr=0.003,
        distil_weight=0.5,
        temperature=10.0,
        num_students=3,
        use_pretrained=False,
        use_scheduler=True,
        use_weighted_dl=False,
        schedule_distil_weight=False,
        seed=42
    )

    main(
        "dml",
        5,
        100,
        1024,
        "/data1/9cuk/kd_lib/hyperparams/adamw-003-10-True",
        loss_fn=CustomKLDivLoss(apply_softmax=True), # only used for DML
        lr=0.003,
        distil_weight=0.5,
        temperature=10.0,
        num_students=3,
        use_pretrained=False,
        use_scheduler=True,
        use_weighted_dl=False,
        schedule_distil_weight=True,
        seed=42
    )
