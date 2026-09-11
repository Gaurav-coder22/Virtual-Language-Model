import os
import sys

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


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

from src.models.transfer_caption_model import TransferCaptioningModel

# ============================================================
# Paths
# ============================================================

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_model_transfer.pth"
)

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "test.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "Images"
)


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
# Load checkpoint
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

word_to_idx = checkpoint["word_to_idx"]
idx_to_word = checkpoint["idx_to_word"]
vocab_size = checkpoint["vocab_size"]


# ============================================================
# Create model
# ============================================================

model = TransferCaptioningModel(
    vocab_size=vocab_size,
    feature_dim=256,
    embed_dim=256,
    hidden_dim=256,
    train_cnn=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load test data
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)

test_images = test_df["image"].unique()

print(
    "\nTest images:",
    len(test_images)
)


# ============================================================
# Special tokens
# ============================================================

start_token = word_to_idx["<START>"]
end_token = word_to_idx["<END>"]


# ============================================================
# BLEU smoothing
# ============================================================

smoother = SmoothingFunction().method1


# ============================================================
# Store scores
# ============================================================

bleu1_scores = []
bleu2_scores = []
bleu4_scores = []


# ============================================================
# Evaluate
# ============================================================

for image_name in tqdm(
    test_images,
    desc="Evaluating test images"
):

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    # --------------------------------------------------------
    # Check image exists
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        print(
            f"\nSkipping missing image: {image_name}"
        )

        continue


    # --------------------------------------------------------
    # Get reference captions
    # --------------------------------------------------------

    references = test_df[
        test_df["image"] == image_name
    ]["caption"].tolist()


    # --------------------------------------------------------
    # Clean references
    # --------------------------------------------------------

    clean_references = []

    for reference in references:

        words = reference.split()

        words = [
            word
            for word in words
            if word not in [
                "<START>",
                "<END>",
                "<PAD>"
            ]
        ]

        clean_references.append(
            words
        )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    # --------------------------------------------------------
    # Generate caption
    # --------------------------------------------------------

    with torch.no_grad():

        generated = model.generate(
            image_tensor,
            start_token=start_token,
            end_token=end_token,
            max_length=30
        )


    # --------------------------------------------------------
    # Convert IDs to words
    # --------------------------------------------------------

    generated_tokens = generated[
        0
    ].tolist()

    generated_words = []

    for token in generated_tokens:

        word = idx_to_word[token]

        if word == "<END>":
            break

        if word == "<START>":
            continue

        if word == "<PAD>":
            continue

        generated_words.append(word)


    candidate = generated_words


    # --------------------------------------------------------
    # Skip empty predictions
    # --------------------------------------------------------

    if len(candidate) == 0:
        continue


    # --------------------------------------------------------
    # BLEU-1
    # --------------------------------------------------------

    bleu1 = sentence_bleu(
        clean_references,
        candidate,
        weights=(1, 0, 0, 0),
        smoothing_function=smoother
    )


    # --------------------------------------------------------
    # BLEU-2
    # --------------------------------------------------------

    bleu2 = sentence_bleu(
        clean_references,
        candidate,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoother
    )


    # --------------------------------------------------------
    # BLEU-4
    # --------------------------------------------------------

    bleu4 = sentence_bleu(
        clean_references,
        candidate,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoother
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu4_scores.append(bleu4)


# ============================================================
# Calculate averages
# ============================================================

average_bleu1 = sum(
    bleu1_scores
) / len(bleu1_scores)

average_bleu2 = sum(
    bleu2_scores
) / len(bleu2_scores)

average_bleu4 = sum(
    bleu4_scores
) / len(bleu4_scores)


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Images evaluated: {len(bleu1_scores)}"
)

print(
    f"Average BLEU-1: {average_bleu1:.4f}"
)

print(
    f"Average BLEU-2: {average_bleu2:.4f}"
)

print(
    f"Average BLEU-4: {average_bleu4:.4f}"
)

print("=" * 60)