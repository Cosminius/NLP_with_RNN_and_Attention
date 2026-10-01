# NLP with RNNs and Attention

This repository contains my notes and experiments while learning sequence models from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurelien Geron. The examples are implemented with PyTorch and common NLP libraries.

The notebooks are meant for learning. They walk through the full process: preparing data, building a model, training it, and testing its predictions.

## Contents

- [Environment](#environment)
- [Notebook guide](#notebook-guide)
- [Data and model artifacts](#data-and-model-artifacts)
- [Notes and limitations](#notes-and-limitations)

## Environment

The project uses Python 3.13 and `uv` to manage the environment.

```bash
uv sync
source .venv/bin/activate
```

Start Jupyter, or open the notebooks in VS Code with `.venv` selected as the Python environment:

```bash
uv run jupyter lab
```

The project uses PyTorch, TorchMetrics, Hugging Face Datasets and Transformers, Tokenizers, scikit-learn, pandas, matplotlib, KaggleHub, and Requests. A GPU is useful for the larger notebooks. Several notebooks use `cuda` directly, so change the `device` value if you are running on a CPU-only machine.

## Notebook guide

### `RNN_with_Attention.ipynb`

This notebook starts with Shakespeare text and builds a character-level sequence model. It covers downloading the text, creating a character vocabulary, encoding and decoding text, preparing batches, and working with recurrent models. It also introduces attention and variable-length sequences.

**Main concepts:** character tokenization, embeddings, RNN/GRU-style sequence processing, padding, packed sequences, attention, and text generation.

### `SentimentAnalysis_with_RNN.ipynb`

This notebook builds a sentiment classifier for the Stanford IMDB dataset. It trains a BPE tokenizer, compares several tokenization methods, pads and truncates reviews, and trains an RNN classifier with PyTorch.

**Main concepts:** binary text classification, BPE, Byte-Level BPE, WordPiece, Unigram tokenization, padding masks, packed sequences, embeddings, and validation metrics.

The tokenizer experiments also show why a decoder is useful when rebuilding text from subword tokens.

### `English-Spanish-NMT.ipynb`

This notebook trains a small English-to-Spanish translation model on the Tatoeba dataset. It starts with a GRU encoder-decoder, uses teacher forcing during training, and generates translations one token at a time. It also explores beam search and attention.

**Main concepts:** sequence-to-sequence learning, BPE tokenization, encoder-decoder GRUs, teacher forcing, autoregressive decoding, beam search, attention, padding, and packed sequences.

Example workflow:

```text
English input -> encoder -> decoder -> Spanish output
```

The notebook downloads data from Hugging Face and can save trained weights locally. The generated model files are ignored by Git.

### `DateConvertor.ipynb`

This notebook converts dates between ISO format and a natural-language format using a GRU encoder-decoder. For example:

```text
2001-01-02 -> January 02, 2001
```

It uses CTA ridership dates downloaded through KaggleHub, trains a small BPE tokenizer, prepares shifted decoder inputs and labels, and evaluates the model with token-level accuracy.

**Main concepts:** tabular data preparation, date formatting, BPE tokenization, teacher forcing, sequence padding, GRU encoder-decoder models, and autoregressive inference.

This is a small experiment. Token-level accuracy is different from exact full-date accuracy, and dates outside the training data may be generated incorrectly. There is also an experimental `date_convertv2` cell that still needs a small fix before it runs.

### `embedded_reber_classification.ipynb`

This notebook generates strings from the Embedded Reber Grammar and trains a recurrent classifier to tell valid and invalid sequences apart. The data is synthetic, so the focus is on sequence memory and classification.

**Main concepts:** finite-state grammars, synthetic sequence generation, variable-length examples, recurrent classification, batching, and sequence padding.

### `test.ipynb`

This contains small PyTorch experiments with tensor shapes, `unsqueeze`, and basic tensor operations. It is scratch material, not part of the main project walkthrough.

## Data and model artifacts

The notebooks use external or generated data:

- Shakespeare text is downloaded from `https://homl.info/shakespeare`.
- IMDB and English-Spanish translation data are loaded through Hugging Face Datasets.
- Date-conversion data is loaded through KaggleHub and may require network access or authentication.
- Embedded Reber examples are generated locally.

Downloaded datasets, local checkpoints, model weights, and experiment outputs are excluded from new Git commits. The repository keeps the notebooks and dependency files so the experiments can be reproduced in a fresh environment.

`uv.lock` is committed to preserve reproducible dependency resolution. The local `.venv` directory is not committed.

## Notes and limitations

- These notebooks are for learning, not production use.
- Most training metrics are token-level metrics, so they do not always show whether a complete translation or date is correct.
- Tokenizers trained with a whitespace pre-tokenizer can split punctuation and subwords in ways that make naive string reconstruction look unusual.
- Model quality depends on the downloaded data, tokenizer vocabulary, sequence length, device, and random seed.
- Notebook execution order matters because definitions and trained objects are kept in the active kernel.
- Some notebooks contain exploratory cells, saved outputs, or small known errors. They are study material rather than polished pipelines.

## Attribution

This is a personal study project inspired by the sequence-modeling material in Aurelien Geron's *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. The neural-network examples use PyTorch.