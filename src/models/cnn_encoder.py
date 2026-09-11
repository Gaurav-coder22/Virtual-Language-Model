import torch
import torch.nn as nn


class CNNEncoder(nn.Module):

    def __init__(self, feature_dim=256):
        super().__init__()

        self.features = nn.Sequential(

            # 224 x 224
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 112 x 112
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 56 x 56
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 28 x 28
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # Instead of depending on the exact spatial size,
            # reduce the feature map to 1 x 1.
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(
            256,
            feature_dim
        )

    def forward(self, images):

        x = self.features(images)

        # [batch_size, 256, 1, 1]
        x = x.view(
            x.size(0),
            -1
        )

        # [batch_size, feature_dim]
        x = self.fc(x)

        return x
