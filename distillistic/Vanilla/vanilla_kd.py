import torch.nn.functional as F

from .base_class import BaseClass


class VanillaKD(BaseClass):
    """
    Original implementation of Knowledge distillation from the paper "Distilling the
    Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    :param use_amp (bool): True to use Automated Mixed Precision
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        loss_fn=None,
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./experiments",
        use_amp=False,
    ):
        if loss_fn is not None:
            print("The argument loss_fn is deprecated. The loss is calculated internally.")

        super(VanillaKD, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
            use_amp,
        )

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation.
        Note that self.loss_fn is ignored, as this implementation is specific to KLD

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        soft_teacher_out = F.softmax(y_pred_teacher / self.temp, dim=1)
        soft_student_out = F.log_softmax(y_pred_student / self.temp, dim=1)

        supervised = F.cross_entropy(y_pred_student, y_true)
        distillation = (self.temp ** 2) * F.kl_div(input=soft_student_out,
                                                   target=soft_teacher_out,
                                                   reduction='batchmean', log_target=False)
        loss = (1 - self.distil_weight) * supervised + self.distil_weight * distillation

        return loss, supervised, distillation
