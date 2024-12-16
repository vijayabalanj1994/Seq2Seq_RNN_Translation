# Seq2Seq Translation Model with PyTorch

This repository contains an implementation of a Seq2Seq model for machine translation, built using PyTorch. The project involves training a neural network to translate sentences from German to English, with functionalities for model evaluation and BLEU score computation.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

This project demonstrates the development of a sequence-to-sequence (Seq2Seq) model for translating sentences. The model is trained on German-English sentence pairs and supports evaluation of translation quality using the BLEU score metric.

### Key Components
- **Encoder**: Processes input sentences and generates context vectors.
- **Decoder**: Generates translated sentences using the context vectors.

## Features

- Preprocessing pipeline for German-English sentence pairs.
- Seq2Seq model architecture with LSTM-based encoder and decoder.
- Visualization of training and validation loss/perplexity.
- BLEU score computation for evaluating translation quality.
- End-to-end implementation using PyTorch.

## Usage

### Training

Train the Seq2Seq model on your dataset:
```bash
python train.py --data_path data/ --epochs 10 --batch_size 64
```

### Translation

Generate translations for German sentences:
```python
from model import generate_translation

src_sentence = "Ein asiatischer Mann kehrt den Gehweg."
translation = generate_translation(model, src_sentence, vocab_transform['de'], vocab_transform['en'])
print("Translated Sentence:", translation)
```

### BLEU Score Evaluation

Evaluate the quality of the generated translations using the BLEU score:
```python
reference_translations = [
    "An Asian man is sweeping the sidewalk.",
    "An Asian man sweeps the walkway."
]
bleu_score = calculate_bleu_score(translation, reference_translations)
print("BLEU Score:", bleu_score)
```

## Results

- Training and validation loss/perplexity are visualized during training.
- The generated translations are evaluated using the BLEU score metric.

## Future Work

- Add support for other language pairs.
- Experiment with different architectures for improved performance.
- Explore transformer-based models for advanced translation tasks.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
