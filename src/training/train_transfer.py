import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

from src.data.dataset import create_dataloader
from src.data.vocabulary import Vocabulary
from src.models.transfer_caption_model import TransferCaptioningModel


# ============================================================
# Paths
# ============================================================

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train.csv"
)

VAL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "val.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "Images"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model_transfer.pth"
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32

# Keep this at 1 for the first test
EPOCHS = 6

FEATURE_DIM = 256
EMBED_DIM = 256
HIDDEN_DIM = 256

LEARNING_RATE = 0.001


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Load datasets
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)

print(
    "Training captions:",
    len(train_df)
)

print(
    "Validation captions:",
    len(val_df)
)

print(
    "Training images:",
    train_df["image"].nunique()
)

print(
    "Validation images:",
    val_df["image"].nunique()
)


# ============================================================
# Build vocabulary
# ============================================================

print("\nBuilding vocabulary...")

vocabulary = Vocabulary(
    min_freq=5
)

vocabulary.build(
    train_df["caption"].tolist()
)

print(
    "Vocabulary size:",
    len(vocabulary)
)


# ============================================================
# Create DataLoaders
# ============================================================

print("\nCreating DataLoaders...")

train_loader = create_dataloader(
    dataframe=train_df,
    image_dir=IMAGE_DIR,
    vocabulary=vocabulary,
    batch_size=BATCH_SIZE,
    shuffle=True,
    train=True
)

val_loader = create_dataloader(
    dataframe=val_df,
    image_dir=IMAGE_DIR,
    vocabulary=vocabulary,
    batch_size=BATCH_SIZE,
    shuffle=False,
    train=False
)

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# Create model
# ============================================================

print("\nCreating transfer-learning model...")

model = TransferCaptioningModel(
    vocab_size=len(vocabulary),
    feature_dim=FEATURE_DIM,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,

    # Freeze pretrained ResNet
    train_cnn=False
)

model = model.to(device)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)


# ============================================================
# Optimizer
# ============================================================

# ResNet is frozen.
# Only the projection layer and LSTM are trained.

optimizer = torch.optim.Adam(
    filter(
        lambda parameter:
        parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


# ============================================================
# Best validation loss
# ============================================================

best_val_loss = float("inf")


# ============================================================
# Training loop
# ============================================================

for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    # Keep frozen ResNet in evaluation mode
    model.encoder.cnn.eval()

    total_train_loss = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"
    )

    for images, captions in train_progress:

        # Move to GPU
        images = images.to(
            device,
            non_blocking=True
        )

        captions = captions.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Teacher forcing
        # ----------------------------------------------------

        inputs = captions[:, :-1]

        targets = captions[:, 1:]

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images,
            inputs
        )

        # ----------------------------------------------------
        # Reshape
        # ----------------------------------------------------

        outputs = outputs.reshape(
            -1,
            len(vocabulary)
        )

        targets = targets.reshape(
            -1
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            targets
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Track loss
        # ----------------------------------------------------

        total_train_loss += loss.item()

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = (
        total_train_loss
        / len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_val_loss = 0.0

    val_progress = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"
    )

    with torch.no_grad():

        for images, captions in val_progress:

            images = images.to(
                device,
                non_blocking=True
            )

            captions = captions.to(
                device,
                non_blocking=True
            )

            inputs = captions[:, :-1]

            targets = captions[:, 1:]

            outputs = model(
                images,
                inputs
            )

            outputs = outputs.reshape(
                -1,
                len(vocabulary)
            )

            targets = targets.reshape(
                -1
            )

            loss = criterion(
                outputs,
                targets
            )

            total_val_loss += loss.item()

            val_progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )


    val_loss = (
        total_val_loss
        / len(val_loader)
    )


    # ========================================================
    # Print results
    # ========================================================

    print("\n" + "=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )

    print("=" * 60)


    # ========================================================
    # Save best model
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "word_to_idx":
                    vocabulary.word_to_idx,

                "idx_to_word":
                    vocabulary.idx_to_word,

                "vocab_size":
                    len(vocabulary),

                "epoch":
                    epoch + 1,

                "val_loss":
                    val_loss
            },
            CHECKPOINT_PATH
        )

        print(
            "Best transfer model saved!"
        )

        print(
            f"Validation loss: {val_loss:.4f}"
        )

    else:

        print(
            "Validation loss did not improve."
        )


# ============================================================
# Finished
# ============================================================

print("\nTraining finished.")

print(
    "Best validation loss:",
    f"{best_val_loss:.4f}"
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)