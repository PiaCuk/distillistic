import os

from distillistic import distillation_experiment, test_distiller, CustomKLDivLoss, SoftKLDivLoss

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dataset_path = "/home/pia/Documents/ImageNet"
    save_path = "/home/pia/Documents/distillistic/experiments"

    # Use new universal main
    for algo in ["dml", "dml_e", "tfkd", "vanilla"]:
        params = {
            "algo": algo,
            "runs": 5,
            "epochs": 100,
            "batch_size": 1024,
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
        distillation_experiment(**params)