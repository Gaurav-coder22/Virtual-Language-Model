import os
import sys

import pandas as pd
import torch
from torch.optim import Adam
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

from src.data.vocabulary import Vocabulary
from src.data.dataset import create_dataloader
from src.models.caption_model import ImageCaptioningModel
from src.training.losses import get_caption_loss


# ============================================================
# Configuration
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

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.001

MIN_FREQ = 5

FEATURE_DIM = 256

EMBED_DIM = 256

HIDDEN_DIM = 256


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
# Load train and validation data
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
# Build vocabulary ONLY from training captions
# ============================================================

print("\nBuilding vocabulary...")

vocabulary = Vocabulary(
    min_freq=MIN_FREQ
)

vocabulary.build(
    train_df["caption"]
)

vocab_size = len(vocabulary)

print(
    "Vocabulary size:",
    vocab_size
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

print("\nCreating model...")

model = ImageCaptioningModel(
    vocab_size=vocab_size,
    feature_dim=FEATURE_DIM,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
)

model = model.to(device)


# ============================================================
# Loss
# ============================================================

criterion = get_caption_loss()


# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Checkpoint directory
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
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
    # TRAINING
    # ========================================================

    model.train()

    total_train_loss = 0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"
    )

    for images, captions in train_progress:

        # ----------------------------------------------
        # Move data to GPU
        # ----------------------------------------------

        images = images.to(
            device,
            non_blocking=True
        )

        captions = captions.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------
        # Teacher forcing
        # ----------------------------------------------

        inputs = captions[:, :-1]

        targets = captions[:, 1:]

        # ----------------------------------------------
        # Clear gradients
        # ----------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------
        # Forward pass
        # ----------------------------------------------

        outputs = model(
            images,
            inputs
        )

        # ----------------------------------------------
        # Reshape
        # ----------------------------------------------

        outputs = outputs.reshape(
            -1,
            vocab_size
        )

        targets = targets.reshape(
            -1
        )

        # ----------------------------------------------
        # Loss
        # ----------------------------------------------

        loss = criterion(
            outputs,
            targets
        )

        # ----------------------------------------------
        # Backpropagation
        # ----------------------------------------------

        loss.backward()

        # ----------------------------------------------
        # Update weights
        # ----------------------------------------------

        optimizer.step()

        # ----------------------------------------------
        # Track loss
        # ----------------------------------------------

        total_train_loss += loss.item()

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    # Average training loss

    train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"
        )

        for images, captions in val_progress:

            # ------------------------------------------
            # Move to GPU
            # ------------------------------------------

            images = images.to(
            device,
            non_blocking=True
        )

            captions = captions.to(
            device,
            non_blocking=True
        )

            # ------------------------------------------
            # Teacher forcing
            # ------------------------------------------

            inputs = captions[:, :-1]

            targets = captions[:, 1:]

            # ------------------------------------------
            # Forward pass
            # ------------------------------------------

            outputs = model(
                images,
                inputs
            )

            # ------------------------------------------
            # Reshape
            # ------------------------------------------

            outputs = outputs.reshape(
                -1,
                vocab_size
            )

            targets = targets.reshape(
                -1
            )

            # ------------------------------------------
            # Loss
            # ------------------------------------------

            loss = criterion(
                outputs,
                targets
            )

            total_val_loss += loss.item()

            val_progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    # Average validation loss

    val_loss = (
        total_val_loss /
        len(val_loader)
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
    # Save latest checkpoint
    # ========================================================

    latest_path = os.path.join(
        CHECKPOINT_DIR,
        "latest_model.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "word_to_idx": vocabulary.word_to_idx,
            "idx_to_word": vocabulary.idx_to_word
        },
        latest_path
    )


    # ========================================================
    # Save best model
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "vocab_size": vocab_size,
                "word_to_idx": vocabulary.word_to_idx,
                "idx_to_word": vocabulary.idx_to_word
            },
            best_path
        )

        print(
            f"Best model saved! "
            f"Validation loss: {val_loss:.4f}"
        )

    else:

        print(
            "Validation loss did not improve."
        )