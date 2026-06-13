# Transformer Stock Price Prediction

A Jupyter notebook that builds a Transformer encoder from scratch in TensorFlow/Keras and uses it to forecast synthetic stock prices from past price windows. Includes training callbacks, window-size experiments, and baseline model comparisons.

## Overview

This project walks through:

1. Generating a synthetic stock price time series
2. Normalizing data and creating sliding-window sequences
3. Implementing multi-head self-attention and Transformer blocks
4. Training a compact Transformer encoder with early stopping and learning-rate scheduling
5. Comparing different lookback windows (`time_step`)
6. Benchmarking against an LSTM and a naive last-price baseline
7. Evaluating and plotting predictions on a held-out test set

## Project Structure

```
Transformer/
├── README.md
├── transformer.ipynb          # Main notebook
└── synthetic_stock_data.csv   # Generated dataset (created by the notebook)
```

## Requirements

- Python 3.10+
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

## How to Run

1. Open `transformer.ipynb` in Jupyter.
2. Run all cells from top to bottom.
3. The notebook will:
   - Create `synthetic_stock_data.csv` (2,000 price points)
   - Build sequences and train the Transformer with early stopping
   - Compare lookback windows, LSTM, and a naive baseline
   - Plot metrics and predictions for all models

### Fast mode (default)

Cell 2 sets `FAST_MODE = True` for quicker CPU runs:

- Smaller model (`embed_dim=32`, `num_layers=1`)
- Shorter lookback window (`time_step=60`)
- Fewer epochs with tighter early stopping
- Caps training samples at 1,000
- Compares only 2 window sizes instead of 3

Set `FAST_MODE = False` in cell 2 for fuller experiments. Skip cell 10 if you only need the main Transformer result.

## Model Architecture

| Component | Value |
|-----------|-------|
| Encoder layers | 2 |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Feed-forward dimension | 128 |
| Default input window | 100 timesteps |
| Output | 1 (next scaled price) |
| Pooling | `GlobalAveragePooling1D` |
| Loss | MSE |
| Optimizer | Adam |

### Training Callbacks

- **EarlyStopping** — stops when `val_loss` stops improving (patience 5), restores best weights
- **ReduceLROnPlateau** — halves the learning rate when `val_loss` plateaus (patience 3)

The model uses a time-ordered 80/20 train/test split so evaluation reflects forecasting on future data rather than random shuffling.

## Data Pipeline

- **Raw data:** Synthetic prices with an upward trend plus Gaussian noise
- **Scaling:** `MinMaxScaler` to range `[0, 1]`
- **Sequences:** Each sample uses consecutive prices (`X`) to predict the following price (`Y`)
- **Default shapes:** `X` → `(1899, 100, 1)`, `Y` → `(1899,)`

## Notebook Cells

| Cell | Purpose |
|------|---------|
| 0 | Imports |
| 1 | Generate synthetic stock data |
| 2 | Load, scale, sequence helpers, and default `time_step=100` |
| 3–5 | Transformer building blocks (attention, blocks, encoder layer) |
| 6 | Full `TransformerEncoder` definition and smoke test |
| 7 | Model builders (Transformer, LSTM) and training callbacks |
| 8 | Train Transformer with early stopping and LR scheduling |
| 9 | Evaluate Transformer and plot test predictions |
| 10 | Compare `time_step` windows: 50, 100, 150 |
| 11 | Train LSTM baseline; compute naive last-price baseline |
| 12 | Side-by-side model comparison (metrics + plots) |

## Extensions Implemented

- Early stopping and learning-rate scheduling
- Lookback window comparison (`50`, `100`, `150`)
- LSTM baseline and naive last-price baseline
- RMSE / MAE / MSE metrics across all models

## Notes

- Predictions are on **scaled** values during training; evaluation inverse-transforms them back to the original price scale.
- The first `model.predict()` call can be slower due to TensorFlow graph compilation.
- Cells 10–12 train additional models and will take longer than the base Transformer run.
- For faster runs on CPU, keep the smaller model settings in `build_transformer_model` (`embed_dim=64`, `num_layers=2`).
