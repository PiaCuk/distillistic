import torch

from distillistic.DML import DML
from distillistic.Vanilla import VanillaKD
from distillistic.Tf_KD import VirtualTeacher
from distillistic.models import resnet18, resnet50
from distillistic.utils import CustomKLDivLoss


def _create_optim(params, lr, adam=True):
    if adam:
        return torch.optim.AdamW(params, lr, betas=(0.9, 0.999))
    else:
        # Zhang et al. use no weight decay and nesterov=True
        return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=0.0001)


def create_distiller(
    algo,
    train_loader,
    test_loader,
    device,
    save_path,
    num_classes,
    loss_fn=CustomKLDivLoss(),
    lr=0.01,
    distil_weight=0.5,
    temperature=10.0,
    num_students=2,
    pretrained=False,
    use_adam=True):
    """
    Create distillers for benchmarking.

    :param algo (str): Name of the training algorithm to use. Either "dml", "dml_e", else VanillaKD
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param test_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param device (str): Device used for training
    :param save_path (str): Directory for storing logs and saving models
    :param num_classes(int): Number of classes to predict
    :param loss_fn (torch.nn.Module): Loss Function used for distillation. Not used for VanillaKD (BaseClass), as it is implemented internally
    :param lr (float): Learning rate
    :param distil_weight (float): Between 0 and 1
    :param temperature (float): temperature parameter for soft targets
    :param num_students (int): Number of students in cohort. Used for DML
    :param pretrained (bool): True to use pretrained torchvision models
    :param use_adam (bool): True to use Adam optim
    """
    student_params = {"num_classes": num_classes, "pretrained": pretrained, "last_layer_only": pretrained}

    if algo == "dml" or algo == "dml_e":
        # Define models
        student_cohort = [resnet18(**student_params) for i in range(num_students)]
        student_optimizers = [_create_optim(
            student_cohort[i].parameters(), lr, adam=use_adam) for i in range(num_students)]
        # Define DML with logging to Tensorboard
        distiller = DML(student_cohort, train_loader, test_loader, student_optimizers, loss_fn=loss_fn, distil_weight=distil_weight,
                        log=True, logdir=save_path, device=device, use_ensemble=True if algo == "dml_e" else False)
    elif algo == "tfkd":
        student = resnet18(**student_params)
        student_optimizer = _create_optim(
            student.parameters(), lr, adam=use_adam)
        # Define TfKD with logging to Tensorboard
        distiller = VirtualTeacher(student, train_loader, test_loader, student_optimizer,
            temp=temperature, distil_weight=distil_weight, log=True, logdir=save_path, device=device)
    else:
        teacher = resnet50(num_classes=num_classes, pretrained=True)
        student = resnet18(**student_params)

        teacher_optimizer = _create_optim(
            teacher.parameters(), lr, adam=use_adam)
        student_optimizer = _create_optim(
            student.parameters(), lr, adam=use_adam)
        # Define KD with logging to Tensorboard
        distiller = VanillaKD(teacher, student, train_loader, test_loader, teacher_optimizer, student_optimizer,
                              temp=temperature, distil_weight=distil_weight, log=True, logdir=save_path, device=device)

    return distiller