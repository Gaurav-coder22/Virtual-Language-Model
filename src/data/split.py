import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import preprocess_captions


def split_dataset(
    df,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42
):
    """
    Split Flickr8k by IMAGE, not by caption.

    This prevents captions belonging to the same image
    from appearing in different splits.
    """

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        )

    # Get unique image names
    images = df["image"].unique()

    # First: train vs temporary
    train_images, temp_images = train_test_split(
        images,
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )

    # Second: validation vs test
    relative_test_ratio = (
        test_ratio / (val_ratio + test_ratio)
    )

    val_images, test_images = train_test_split(
        temp_images,
        test_size=relative_test_ratio,
        random_state=random_state
    )

    # Convert to sets for fast lookup
    train_images = set(train_images)
    val_images = set(val_images)
    test_images = set(test_images)

    # Create DataFrames
    train_df = df[
        df["image"].isin(train_images)
    ].reset_index(drop=True)

    val_df = df[
        df["image"].isin(val_images)
    ].reset_index(drop=True)

    test_df = df[
        df["image"].isin(test_images)
    ].reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":

    # --------------------------------
    # Paths
    # --------------------------------

    caption_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "captions.txt"
    )

    processed_dir = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed"
    )

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    # --------------------------------
    # Load captions
    # --------------------------------

    print("Loading captions...")

    df = preprocess_captions(
        caption_file
    )

    print(
        "Total captions:",
        len(df)
    )

    print(
        "Total images:",
        df["image"].nunique()
    )

    # --------------------------------
    # Split
    # --------------------------------

    train_df, val_df, test_df = split_dataset(df)

    # --------------------------------
    # Save
    # --------------------------------

    train_path = os.path.join(
        processed_dir,
        "train.csv"
    )

    val_path = os.path.join(
        processed_dir,
        "val.csv"
    )

    test_path = os.path.join(
        processed_dir,
        "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------
    # Results
    # --------------------------------

    print("\nSplit completed!")

    print(
        f"Train: {len(train_df)} captions "
        f"from {train_df['image'].nunique()} images"
    )

    print(
        f"Validation: {len(val_df)} captions "
        f"from {val_df['image'].nunique()} images"
    )

    print(
        f"Test: {len(test_df)} captions "
        f"from {test_df['image'].nunique()} images"
    )

    # --------------------------------
    # Check for leakage
    # --------------------------------

    train_set = set(train_df["image"])
    val_set = set(val_df["image"])
    test_set = set(test_df["image"])

    print(
        "\nTrain ∩ Validation:",
        len(train_set & val_set)
    )

    print(
        "Train ∩ Test:",
        len(train_set & test_set)
    )

    print(
        "Validation ∩ Test:",
        len(val_set & test_set)
    )