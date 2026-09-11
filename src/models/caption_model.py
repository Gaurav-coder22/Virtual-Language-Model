import torch
import torch.nn as nn
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.models.cnn_encoder import CNNEncoder
from src.models.lstm_decoder import LSTMDecoder


class ImageCaptioningModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        feature_dim=256,
        embed_dim=256,
        hidden_dim=256,
        num_layers=1
    ):
        super().__init__()

        # -----------------------------
        # CNN Encoder
        # -----------------------------

        self.encoder = CNNEncoder(
            feature_dim=feature_dim
        )

        # -----------------------------
        # LSTM Decoder
        # -----------------------------

        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            num_layers=num_layers
        )

    def forward(self, images, captions):

        # Image
        # ↓
        # CNN
        image_features = self.encoder(images)

        # Image features + captions
        # ↓
        # LSTM
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

        # Extract image features
        image_features = self.encoder(
            images
        )

        # Generate caption
        captions = self.decoder.generate(
            image_features,
            start_token=start_token,
            end_token=end_token,
            max_length=max_length
        )

        return captions

