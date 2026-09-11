import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=256,
        feature_dim=256,
        num_layers=1
    ):
        super().__init__()

        # Convert word IDs into word embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Convert LSTM hidden state to vocabulary scores
        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

        # Convert CNN image features into
        # initial LSTM hidden state
        self.init_hidden = nn.Linear(
            feature_dim,
            hidden_dim * num_layers
        )

        # Convert CNN image features into
        # initial LSTM cell state
        self.init_cell = nn.Linear(
            feature_dim,
            hidden_dim * num_layers
        )

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, image_features, captions):

        # --------------------------------
        # Convert words to embeddings
        # --------------------------------

        embeddings = self.embedding(captions)

        # --------------------------------
        # Create initial hidden state
        # from image features
        # --------------------------------

        hidden = self.init_hidden(
            image_features
        )

        cell = self.init_cell(
            image_features
        )

        # [batch, num_layers * hidden_dim]
        # →
        # [num_layers, batch, hidden_dim]

        hidden = hidden.view(
            image_features.size(0),
            self.num_layers,
            self.hidden_dim
        )

        hidden = hidden.permute(
            1, 0, 2
        ).contiguous()

        cell = cell.view(
            image_features.size(0),
            self.num_layers,
            self.hidden_dim
        )

        cell = cell.permute(
            1, 0, 2
        ).contiguous()

        # --------------------------------
        # LSTM
        # --------------------------------

        output, _ = self.lstm(
            embeddings,
            (hidden, cell)
        )

        # --------------------------------
        # Predict vocabulary
        # --------------------------------

        output = self.fc(output)

        return output
    def generate(
        self,
        image_features,
        start_token,
        end_token,
        max_length=30
    ):
        """
        Generate a caption one word at a time.
        """

        batch_size = image_features.size(0)

        # Create initial hidden state
        hidden = self.init_hidden(image_features)

        cell = self.init_cell(image_features)

        hidden = hidden.view(
            batch_size,
            self.num_layers,
            self.hidden_dim
        )

        hidden = hidden.permute(
            1, 0, 2
        ).contiguous()

        cell = cell.view(
            batch_size,
            self.num_layers,
            self.hidden_dim
        )

        cell = cell.permute(
            1, 0, 2
        ).contiguous()

        # Start with <START>
        current_token = torch.tensor(
            [start_token] * batch_size,
            dtype=torch.long,
            device=image_features.device
        )

        generated_tokens = []

        for _ in range(max_length):

            # [batch] → [batch, 1]
            current_token = current_token.unsqueeze(1)

            # Word embedding
            embedding = self.embedding(
                current_token
            )

            # LSTM
            output, (hidden, cell) = self.lstm(
                embedding,
                (hidden, cell)
            )

            # Vocabulary scores
            output = self.fc(output)

            # [batch, 1, vocab_size]
            # → [batch, vocab_size]
            output = output.squeeze(1)

            # Select highest probability word
            predicted_token = output.argmax(
                dim=1
            )

            generated_tokens.append(
                predicted_token
            )

            # Stop if <END> is generated
            if torch.all(
                predicted_token == end_token
            ):
                break

            current_token = predicted_token

        # [sequence, batch]
        generated_tokens = torch.stack(
            generated_tokens,
            dim=1
        )

        return generated_tokens

