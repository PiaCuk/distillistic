import os

from distillistic import (CustomKLDivLoss, ImageNet_experiment)

# TODO main for ImageNet experiment
# First experiment to test the setup:
# Vanilla KD with pretrained ResNet-50 Teacher and ResNet-18 student
# Data augmentation as in torch example?
# How many epochs?
# 1 run to start

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dataset_path = "/home/pia/Documents/ImageNet"
    save_path = "/home/pia/Documents/distillistic/experiments/test"

    # Use new universal main
    for algo in ["dml", "dml_e", "tfkd", "vanilla"]:
        params = {
            "algo": algo,
            "runs": 1,
            "epochs": 1,
            "batch_size": 4,
            "save_path": save_path,
            "loss_fn": CustomKLDivLoss(apply_softmax=algo != "dml_e"),
            "lr": 0.005,
            "distil_weight": 0.5,
            "temperature": 10.0 if algo != "tfkd" else 1.0,
            "num_students": 3,
            "use_pretrained": True,
            "use_scheduler": True,
            "use_weighted_dl": False,
            "schedule_distil_weight": False,
            "seed": 42,
        }
        ImageNet_experiment(**params)
