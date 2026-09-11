import torch
import torch.nn as nn
from torchvision import models


class PretrainedResNetEncoder(nn.Module):

    def __init__(
        self,
        feature_dim=256,
        train_cnn=False
    ):
        super().__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.DEFAULT

        resnet = models.resnet50(
            weights=weights
        )

        # Remove the final classification layer
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # ResNet50 outputs 2048 features
        self.feature_projection = nn.Linear(
            2048,
            feature_dim
        )

        # Freeze CNN initially
        if not train_cnn:

            for param in self.cnn.parameters():
                param.requires_grad = False


    def forward(self, images):

        # Extract visual features
        features = self.cnn(
            images
        )

        # [batch, 2048, 1, 1]
        features = features.view(
            features.size(0),
            -1
        )

        # 2048 → 256
        features = self.feature_projection(
            features
        )

        return features


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    model = PretrainedResNetEncoder(
        feature_dim=256,
        train_cnn=False
    )

    model = model.to(device)

    x = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    with torch.no_grad():

        output = model(x)

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )