import os

from distillistic import (
    CustomKLDivLoss, ImageNet_experiment, FMNIST_experiment)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dataset = "imagenet"

    if dataset == "imagenet" or dataset == "ffcv-imagenet":
        classes = 1000
    elif dataset == "imagecat" or dataset == "Fashion-MNIST":
        classes = 10
    else:
        classes = None

    dataset_path = f"../data/{dataset}"
    save_path = f"./experiments/{dataset}/cross_session0"

    # Use new universal main
    for idx, scale in enumerate([42, 84, 168]):
        for algo in ["vanilla", "baseline", "dml"]:  # "dml", "dml_e", "tfkd", "vanilla"
            # Already ran sessions 0-2
            save_path = f"./experiments/{dataset}/cross_session{3+idx}"
            params = {
                "algo": algo,
                "runs": 1,
                "epochs": 20,
                "batch_size": 1024 if scale < 120 else 512,
                "data_path": dataset_path,
                "save_path": save_path,
                "loss_fn": CustomKLDivLoss(apply_softmax=algo != "dml_e"),
                "lr": 0.001,
                "distil_weight": 0.5,
                "temperature": 20.0 if algo != "tfkd" else 1.0,
                "num_students": 3,
                "use_pretrained": False,
                "use_scheduler": True,
                "use_weighted_dl": False,
                "schedule_distil_weight": False,
                "seed": 42,
                "classes": classes,
                "use_amp": True,
                "use_ffcv": dataset == "ffcv-imagenet",
                "downscale": (scale, scale),
            }

            if dataset == "Fashion-MNIST":
                FMNIST_experiment(**params)
            else:
                ImageNet_experiment(**params)
    
    for idx, scale in enumerate([42, 84, 168]):
        save_path = f"./experiments/{dataset}/cross_session{3+idx}"
        params = {
            "algo": "vanilla",
            "runs": 1,
            "epochs": 20,
            "batch_size": 1024 if scale < 120 else 512,
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
            "use_amp": True,
            "use_ffcv": dataset == "ffcv-imagenet",
            "downscale": (scale, scale),
        }

        if dataset == "Fashion-MNIST":
            FMNIST_experiment(**params)
        else:
            ImageNet_experiment(**params)
