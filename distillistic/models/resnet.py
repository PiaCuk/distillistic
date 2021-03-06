from torch import nn
from torchvision import models


class CustomResNet(nn.Module):
    """
    Wrapper for ResNets from torchvision.models
    See https://github.com/Stevellen/ResNet-Lightning/blob/master/resnet_classifier.py
    """

    def __init__(self, num_classes, resnet_version, pretrained=False, last_layer_only=False) -> None:
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=pretrained)
        if (num_classes != 1000):
            # Replace old FC layer with Identity so we can train our own
            print(f"Creating new output layer with {num_classes} classes.")
            linear_size = list(self.resnet_model.children())[-1].in_features
            # replace final layer for fine tuning
            self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if last_layer_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)


def resnet18(num_classes, pretrained=False, last_layer_only=False):
    return CustomResNet(num_classes=num_classes,
                        resnet_version=18,
                        pretrained=pretrained,
                        last_layer_only=last_layer_only
                        )


def resnet50(num_classes, pretrained=False, last_layer_only=False):
    return CustomResNet(num_classes=num_classes,
                        resnet_version=50,
                        pretrained=pretrained,
                        last_layer_only=last_layer_only
                        )
