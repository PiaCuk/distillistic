import os

from distillistic import (
    CustomKLDivLoss, ImageNet_experiment, FMNIST_experiment)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    dataset = "Fashion-MNIST"

    if dataset == "imagenet":
        classes = 1000
    elif dataset == "imagecat" or dataset == "Fashion-MNIST":
        classes = 10
    else:
        classes = None

    dataset_path = f"../data/{dataset}"
    save_path = f"./experiments/{dataset}/debug"

    # Use new universal main
    for algo in ["dml"]:  # "dml", "dml_e", "tfkd", "vanilla"
        params = {
            "algo": algo,
            "runs": 1,
            "epochs": 10,
            "batch_size": 1024,
            "data_path": dataset_path,
            "save_path": save_path,
            "loss_fn": CustomKLDivLoss(apply_softmax=algo != "dml_e"),
            "lr": 0.001,
            "distil_weight": 0.5,
            "temperature": 10.0 if algo != "tfkd" else 1.0,
            "num_students": 3,
            "use_pretrained": False,
            "use_scheduler": True,
            "use_weighted_dl": False,
            "schedule_distil_weight": False,
            "seed": 42,
            "classes": classes,
        }

        if dataset == "Fashion-MNIST":
            FMNIST_experiment(**params)
        else:
            ImageNet_experiment(**params)
