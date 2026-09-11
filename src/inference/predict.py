import os
import sys
import random
from urllib.parse import quote

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


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
# Project import
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
# Configuration
# ============================================================

NUM_IMAGES = 5

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
# Load checkpoint
# ============================================================

print("\nLoading transfer-learning checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

word_to_idx = checkpoint[
    "word_to_idx"
]

idx_to_word = checkpoint[
    "idx_to_word"
]

vocab_size = checkpoint[
    "vocab_size"
]


print(
    "Vocabulary size:",
    vocab_size
)

print(
    "Best epoch:",
    checkpoint.get(
        "epoch",
        "unknown"
    )
)

print(
    "Validation loss:",
    checkpoint.get(
        "val_loss",
        "unknown"
    )
)


# ============================================================
# Create model
# ============================================================

model = TransferCaptioningModel(
    vocab_size=vocab_size,
    feature_dim=FEATURE_DIM,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    train_cnn=False
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(device)

model.eval()


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# Load test dataset
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)

test_images = (
    test_df["image"]
    .unique()
    .tolist()
)

print(
    "\nTotal test images:",
    len(test_images)
)


# ============================================================
# Select random test images
# ============================================================

num_to_select = min(
    NUM_IMAGES,
    len(test_images)
)

selected_images = random.sample(
    test_images,
    num_to_select
)


# ============================================================
# Special tokens
# ============================================================

start_token = word_to_idx[
    "<START>"
]

end_token = word_to_idx[
    "<END>"
]


# ============================================================
# Generate captions
# ============================================================

print("\n")
print("=" * 80)
print("TRANSFER LEARNING IMAGE CAPTIONING RESULTS")
print("=" * 80)


for i, image_name in enumerate(
    selected_images,
    start=1
):

    # --------------------------------------------------------
    # Image path
    # --------------------------------------------------------

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )


    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        print(
            f"\nImage not found: {image_path}"
        )

        continue


    # --------------------------------------------------------
    # Create clickable file link
    # --------------------------------------------------------

    photo_link = (
        "file:///"
        + quote(
            image_path.replace(
                "\\",
                "/"
            ),
            safe="/:"
        )
    )


    # --------------------------------------------------------
    # Print image information
    # --------------------------------------------------------

    print("\n")
    print("-" * 80)

    print(
        f"IMAGE {i}/{num_to_select}"
    )

    print("-" * 80)

    print(
        "Filename:",
        image_name
    )

    print(
        "Photo:",
        photo_link
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )


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

        word = idx_to_word[
            token
        ]

        if word == "<END>":
            break

        if word == "<START>":
            continue

        if word == "<PAD>":
            continue

        generated_words.append(
            word
        )


    # --------------------------------------------------------
    # Generated caption
    # --------------------------------------------------------

    generated_caption = " ".join(
        generated_words
    )


    print(
        "\nGenerated Caption:"
    )

    print(
        generated_caption
    )


    # --------------------------------------------------------
    # Reference captions
    # --------------------------------------------------------

    references = test_df[
        test_df["image"] == image_name
    ]["caption"].tolist()


    print(
        "\nHuman Reference Captions:"
    )


    for j, reference in enumerate(
        references,
        start=1
    ):

        reference = (
            reference
            .replace(
                "<START>",
                ""
            )
            .replace(
                "<END>",
                ""
            )
            .replace(
                "<PAD>",
                ""
            )
            .strip()
        )

        print(
            f"{j}. {reference}"
        )


# ============================================================
# Finished
# ============================================================

print("\n")
print("=" * 80)
print("FINISHED")
print("=" * 80)