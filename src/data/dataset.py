import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
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


from src.data.vocabulary import Vocabulary


# ============================================================
# Transforms
# ============================================================

# Used for TRAINING
# Random augmentation helps the CNN generalize better.

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomCrop((224, 224)),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Used for VALIDATION and TESTING
# No random operations here.

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Dataset
# ============================================================

class FlickrDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        vocabulary,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.image_dir = image_dir

        self.vocabulary = vocabulary

        self.transform = transform

        if self.transform is None:
            self.transform = EVAL_TRANSFORM


    def __len__(self):

        return len(
            self.dataframe
        )


    def __getitem__(self, index):

        # ----------------------------------------------------
        # Image filename
        # ----------------------------------------------------

        image_name = self.dataframe.iloc[
            index
        ]["image"]


        # ----------------------------------------------------
        # Image path
        # ----------------------------------------------------

        image_path = os.path.join(
            self.image_dir,
            image_name
        )


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # Transform image
        # ----------------------------------------------------

        image = self.transform(
            image
        )


        # ----------------------------------------------------
        # Caption
        # ----------------------------------------------------

        caption = self.dataframe.iloc[
            index
        ]["caption"]


        # ----------------------------------------------------
        # Convert caption to token IDs
        # ----------------------------------------------------

        caption_sequence = self.vocabulary.numericalize(
            caption
        )

        caption_sequence = torch.tensor(
            caption_sequence,
            dtype=torch.long
        )


        return image, caption_sequence


# ============================================================
# Collate function
# ============================================================

def collate_fn(batch):

    images = []
    captions = []

    for image, caption in batch:

        images.append(image)

        captions.append(caption)


    # --------------------------------------------------------
    # Stack images
    # --------------------------------------------------------

    images = torch.stack(
        images
    )


    # --------------------------------------------------------
    # Pad captions
    # --------------------------------------------------------

    captions = torch.nn.utils.rnn.pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )


    return images, captions


# ============================================================
# DataLoader
# ============================================================

def create_dataloader(
    dataframe,
    image_dir,
    vocabulary,
    batch_size=32,
    shuffle=True,
    train=False
):

    if train:

        transform = TRAIN_TRANSFORM

    else:

        transform = EVAL_TRANSFORM


    dataset = FlickrDataset(
        dataframe=dataframe,
        image_dir=image_dir,
        vocabulary=vocabulary,
        transform=transform
    )


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )


    return dataloader


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print(
        "Train transform:"
    )

    print(
        TRAIN_TRANSFORM
    )

    print(
        "\nEvaluation transform:"
    )

    print(
        EVAL_TRANSFORM
    )