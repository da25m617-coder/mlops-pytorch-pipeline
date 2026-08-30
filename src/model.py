import torch
import torch.nn as nn
from torchvision.models import resnet18


class FashionCNN(nn.Module):
    """
    Small CNN designed for Fashion-MNIST:
    input: 1 x 28 x 28
    output: 10 classes
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 28x28 -> 14x14 -> 7x7
            # 64 feature maps -> 64 * 7 * 7
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10
) -> nn.Module:

    if architecture == "cnn":
        return FashionCNN(num_classes=num_classes)

    if architecture == "resnet18":
        model = resnet18(weights=None)

        # Adapt ResNet-18 for 1-channel 28x28 Fashion-MNIST
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        model.maxpool = nn.Identity()

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}"
    )