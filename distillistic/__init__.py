from .models import (
    CustomResNet,
    resnet18,
    resnet50,
)

from .Vanilla import (
    VanillaKD,
)

from .DML import (
    DML,
)

from .Tf_KD import (
    VirtualTeacher,
)

from .data import (
    FMNIST_loader,
    ImageNet_loader,
)

from .utils import (
    ECELoss,
    CustomKLDivLoss,
    SoftKLDivLoss,
    set_seed,
    create_distiller
)

from .fmnist import (
    distillation_experiment,
    test_distiller,
)