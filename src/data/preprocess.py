import os
import re
import pandas as pd


def clean_caption(caption):
    """
    Clean a single caption.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove extra whitespace
    4. Add start and end tokens
    """

    caption = caption.lower()

    # Keep only letters and spaces
    caption = re.sub(r"[^a-z\s]", "", caption)

    # Remove extra spaces
    caption = re.sub(r"\s+", " ", caption).strip()

    # Add special tokens
    caption = f"<START> {caption} <END>"

    return caption


def load_captions(caption_file):
    """
    Load Flickr8k captions.txt.

    Expected format:

    image_name.jpg,caption text
    """

    data = []

    with open(caption_file, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Skip header if present
            if line.lower().startswith("image,"):
                continue

            image_name, caption = line.split(",", 1)

            data.append({
                "image": image_name,
                "caption": caption
            })

    return pd.DataFrame(data)


def preprocess_captions(caption_file):
    """
    Load and preprocess all captions.
    """

    df = load_captions(caption_file)

    df["caption"] = df["caption"].apply(clean_caption)

    return df


if __name__ == "__main__":

    caption_file = os.path.join(
        "data",
        "raw",
        "captions.txt"
    )

    df = preprocess_captions(caption_file)

    print("Number of caption records:", len(df))

    print("\nFirst 5 records:")
    print(df.head())

    print("\nExample processed caption:")
    print(df.iloc[0]["caption"])