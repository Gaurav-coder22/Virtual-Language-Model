import torch
import torch.nn as nn

from src.models.pretrained_encoder import PretrainedResNetEncoder
from src.models.lstm_decoder import LSTMDecoder


class TransferCaptioningModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        feature_dim=256,
        embed_dim=256,
        hidden_dim=256,
        train_cnn=False
    ):
        super().__init__()

        # ====================================================
        # Pretrained ResNet50 encoder
        # ====================================================

        self.encoder = PretrainedResNetEncoder(
            feature_dim=feature_dim,
            train_cnn=train_cnn
        )

        # ====================================================
        # Existing LSTM decoder
        # ====================================================

        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )


    def forward(
        self,
        images,
        captions
    ):

        # ----------------------------------------------------
        # Extract image features
        # ----------------------------------------------------

        image_features = self.encoder(
            images
        )

        # ----------------------------------------------------
        # Generate caption predictions
        # ----------------------------------------------------

        outputs = self.decoder(
            image_features,
            captions
        )

        return outputs


    def generate(
        self,
        images,
        start_token,
        end_token,
        max_length=30
    ):

        # ----------------------------------------------------
        # Extract image features
        # ----------------------------------------------------

        image_features = self.encoder(
            images
        )

        # ----------------------------------------------------
        # Generate caption
        # ----------------------------------------------------

        captions = self.decoder.generate(
            image_features,
            start_token=start_token,
            end_token=end_token,
            max_length=max_length
        )

        return captions


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = TransferCaptioningModel(
        vocab_size=2648,
        feature_dim=256,
        embed_dim=256,
        hidden_dim=256,
        train_cnn=False
    )

    model = model.to(device)

    images = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    captions = torch.randint(
        0,
        2648,
        (2, 10)
    ).to(device)

    with torch.no_grad():

        output = model(
            images,
            captions
        )

    print(
        "Input image shape:",
        images.shape
    )

    print(
        "Caption input shape:",
        captions.shape
    )

    print(
        "Output shape:",
        output.shape
    )