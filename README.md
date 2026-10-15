# Virtual Language Model — Image Captioning Using Deep Learning

An image captioning system that generates natural-language descriptions of images using **Computer Vision, Transfer Learning, and Natural Language Processing (NLP)**. The project uses a pretrained ResNet50 CNN as the visual encoder and an LSTM network as the caption decoder.

## Overview

Image captioning combines computer vision and natural language generation to describe the content of an image in human-readable text.

This project explores two architectures:

- **CNN-LSTM (Baseline):** A CNN trained from scratch extracts image features, and an LSTM generates captions.
- **ResNet50-LSTM (Transfer Learning):** A pretrained ResNet50 extracts visual features, which are projected into a lower-dimensional representation and passed to an LSTM decoder.

The transfer-learning approach improved caption quality and reduced repetitive caption generation compared with the baseline.

## Features

- Image feature extraction using CNN and pretrained ResNet50.
- Natural-language caption generation using an LSTM decoder.
- Vocabulary construction and caption tokenization.
- Special tokens: `<PAD>`, `<UNK>`, `<START>`, and `<END>`.
- Training, validation, and testing pipeline.
- Model checkpointing and inference on unseen images.
- Evaluation using BLEU-1, BLEU-2, and BLEU-4.
- Comparison between a CNN-LSTM baseline and a transfer-learning model.

## Model Architecture

```text
Input Image
    |
    v
Image Preprocessing
(Resize, Normalize)
    |
    v
ResNet50 Encoder
(Pretrained CNN)
    |
    v
2048-D Visual Features
    |
    v
Linear Projection
2048 -> 256
    |
    v
LSTM Decoder
    |
    v
Word-by-Word Generation
    |
    v
Generated Image Caption
```

The ResNet50 encoder initially remains frozen during training. The feature projection layer and LSTM decoder are trained to generate captions conditioned on the visual features.

## Dataset

**Dataset:** Flickr8k

The dataset contains images paired with human-written captions. Each image has multiple reference captions, allowing the generated descriptions to be evaluated against human descriptions.

| Split | Images | Captions |
|---|---:|---:|
| Training | 6,472 | 32,360 |
| Validation | 809 | 4,045 |
| Testing | 810 | 4,050 |
| **Total** | **8,091** | **40,455** |

Vocabulary size: **2,648 words**

*Note: Dataset files are not included in this repository. Download Flickr8k separately and place the images and caption files in the appropriate data directories.*

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Computer Vision:** Torchvision, ResNet50
- **Sequence Modeling:** LSTM
- **Image Processing:** Pillow (PIL)
- **Evaluation:** NLTK BLEU metrics
- **Dataset Processing:** Pandas, NumPy
- **Hardware Acceleration:** NVIDIA CUDA-compatible GPU

## Evaluation Results

The following results were obtained on the test set of 810 images.

| Metric | CNN-LSTM Baseline | ResNet50-LSTM |
|---|---:|---:|
| BLEU-1 | 0.3347 | **0.5663** |
| BLEU-2 | 0.1762 | **0.3754** |
| BLEU-4 | 0.0576 | **0.1558** |

The ResNet50-LSTM model improved BLEU-4 by approximately **170% relative to the baseline**.

BLEU measures the overlap between generated captions and human reference captions. BLEU-1 evaluates individual-word matches, while BLEU-4 evaluates matches of sequences of up to four consecutive words. These scores are not classification accuracy.

## Example Predictions

Examples of generated captions from the test set:

| Image | Generated Caption |
|---|---|
| Soccer scene | A soccer player in white is kicking the ball. |
| Dog near water | A white dog is running through the water. |
| Outdoor scene | A man in a red shirt is standing in front of a tree. |
| Dog in a field | A brown and white dog is running in a field. |
| Beach scene | A girl in a yellow bathing suit is surfing. |

These examples demonstrate that the transfer-learning model can generate image-dependent descriptions, although it may miss objects or predict incorrect details.

## Project Structure

```text
image-captioning-cnn-lstm/
├── checkpoints/
├── data/
│   ├── processed/
│   └── raw/
│       └── Images/
├── notebooks/
│   └── experimentation.ipynb
├── outputs/
│   ├── captions/
│   └── examples/
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   └── vocabulary.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── inference/
│   │   └── predict.py
│   ├── models/
│   │   ├── caption_model.py
│   │   ├── cnn_encoder.py
│   │   ├── lstm_decoder.py
│   │   ├── pretrained_encoder.py
│   │   └── transfer_caption_model.py
│   ├── training/
│   │   ├── train.py
│   │   └── train_transfer.py
│   └── utils/
│       ├── config.py
│       └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd image-captioning-cnn-lstm
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install a PyTorch build compatible with your system and GPU if it is not already specified correctly in your requirements.

### 4. Prepare the dataset

Download Flickr8k and place the image files and caption annotations in the expected data directories. Run the project's preprocessing workflow to create the training, validation, and test splits.

Check `src/utils/config.py` and the preprocessing scripts for the expected file paths.

## Training

Train the baseline CNN-LSTM model:

```bash
python -m src.training.train
```

Train the transfer-learning model:

```bash
python -m src.training.train_transfer
```

The trained checkpoints are saved in the `checkpoints/` directory according to the training configuration.

## Inference

After training and preparing the required checkpoint, run:

```bash
python -m src.inference.predict
```

The inference script generates captions for test images and displays reference captions for comparison.

## Evaluation

Evaluate the trained model on the test dataset:

```bash
python -m src.evaluation.evaluate
```

The evaluation script reports BLEU-1, BLEU-2, and BLEU-4 scores.

**Important:** Run inference and evaluation with the same model architecture and matching vocabulary used during training.

## Challenges and Learnings

- Building an end-to-end image captioning pipeline.
- Connecting CNN-based visual features with sequential language generation.
- Handling unknown words, padding, and sequence boundaries.
- Identifying repetitive captions produced by the baseline model.
- Applying transfer learning to improve visual representations.
- Comparing model performance using quantitative metrics and qualitative predictions.

## Future Improvements

- Add an attention mechanism to focus on relevant image regions during caption generation.
- Fine-tune selected ResNet50 layers using a smaller learning rate.
- Experiment with beam search during inference.
- Tune decoder dimensions, dropout, and training hyperparameters.
- Explore Transformer-based decoders and modern vision-language models.
- Improve evaluation with additional metrics such as METEOR, ROUGE-L, and CIDEr.

## Key Learning Outcome

This project demonstrates how pretrained computer vision models and recurrent neural networks can work together to convert visual information into natural-language descriptions. Comparing the baseline with transfer learning provided measurable evidence of improved caption generation.

## Author

Developed as a deep learning project exploring image captioning, transfer learning, computer vision, and NLP.
