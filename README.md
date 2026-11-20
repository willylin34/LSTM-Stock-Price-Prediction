# LSTM-Based Stock Price Prediction

A Python machine learning course project exploring next-day stock closing-price prediction with a **simplified Long Short-Term Memory (LSTM) network implemented using NumPy**. The project focuses on understanding recurrent model components, preparing financial time-series data, and evaluating prediction errors.

> **Scope:** This is an educational LSTM-style implementation, **not a fully trained LSTM**. The recurrent gate weights are randomly initialized and held fixed; gradient descent updates only the final prediction layer. The predictions should not be used for investment decisions.

## Project Overview

The program downloads historical daily closing prices for Apple Inc. (`AAPL`) using Yahoo Finance data accessed through `yfinance`. It creates 10-day sequences of closing prices and predicts the next day's closing price using a custom recurrent model.

The pipeline includes:

1. Downloading historical AAPL closing prices starting January 1, 2015.
2. Scaling prices with Min-Max normalization and generating 10-day input windows.
3. Splitting sequences chronologically into 80% training and 20% testing sets.
4. Passing each window through manually implemented recurrent gates and cell-state updates.
5. Training the final output layer with gradient descent over 20 epochs.
6. Computing training and testing Root Mean Squared Error (RMSE), saving predictions, and plotting results.

## Technologies

- **Language:** Python
- **Libraries:** NumPy, pandas, Matplotlib, yfinance
- **Techniques:** Recurrent neural networks, time-series windowing, Min-Max normalization, chronological train/test split, gradient descent, RMSE evaluation

## Model Implementation

The custom `LSTM` class implements the core forward-pass calculations of an LSTM-like recurrent network:

- **Forget gate:** controls retention of the previous cell state.
- **Input gate:** controls incorporation of candidate information.
- **Candidate memory:** uses a `tanh` activation to generate potential cell updates.
- **Output gate:** determines the exposed hidden state.
- **Prediction layer:** maps the final hidden state to one predicted normalized closing price.

During training, **only the prediction layer's weights and bias are updated** using gradient descent. The code does not implement backpropagation through time (BPTT) for the gate parameters.

### Default configuration

| Parameter | Value |
| --- | --- |
| Stock ticker | `AAPL` |
| Data start date | `2015-01-01` |
| Prediction target | Next trading day's closing price |
| Input window | 10 trading days |
| Train/test split | 80% / 20% (chronological) |
| Hidden units | 32 |
| Learning rate | 0.01 |
| Training epochs | 20 |
| Random seed | 42 |

The source file defines these settings near the top, making them easy to adjust.

## Getting Started

### Requirements

- Python 3
- Internet access to download the historical price data

Install the dependencies:

```bash
pip install -r requirements.txt
```

On Windows, you can also use `py -m pip install numpy pandas matplotlib yfinance` if `python` is not recognized.

### Run the project

From the directory containing the Python script, run:

```bash
python "Term Project - LSTM Model.py"
```

On Windows, `py "Term Project - LSTM Model.py"` is an alternative.

The script prints dataset details, per-epoch training errors, and the final training and testing RMSE. It also writes the following files to the **current working directory**:

| Output | Description |
| --- | --- |
| `predictions.csv` | Actual and predicted closing prices for the test samples |
| `experiment_log.csv` | Configuration and training/test RMSE for the current run |
| `predicted_vs_actual.png` | Actual vs. predicted test-set prices |
| `training_error.png` | Scaled training RMSE across epochs |

The default script evaluates **one configuration per run**. To test different settings, modify the parameters near the top of the file and rerun it; note that output files will be overwritten unless you rename or save them separately.

## Evaluation

The model uses **Root Mean Squared Error (RMSE)** to summarize the difference between predicted and actual closing prices. The final training and test RMSE are calculated after converting normalized predictions back into the original closing-price units.

Because the script downloads data without a fixed end date, its dataset and RMSE values can change between runs. This README intentionally does not claim a fixed or reproducible RMSE score.

## Limitations and Future Improvements

This project is intended to demonstrate the mechanics of an LSTM-style model rather than to establish a production-ready forecasting system.

- **Partial training:** Gate weights are fixed; only the final prediction layer is trained. A next step is to implement BPTT or compare with a fully trainable PyTorch LSTM.
- **Data leakage in normalization:** The current code calculates the Min-Max scaling parameters using the complete series before the train/test split. For a valid out-of-sample evaluation, the scaler should be fitted using training-period data only.
- **Moving dataset:** The Yahoo Finance download has no fixed end date, so exact results may vary over time. Pinning a date range and library versions would improve reproducibility.
- **Limited benchmarking:** Add a simple persistence baseline (e.g., predict the next closing price using the latest observed closing price) before drawing conclusions about forecasting performance.
- **Single feature:** The current model uses closing price alone; additional features and robust validation could be explored.

## Repository Files

```text
LSTM-Stock-Price-Prediction/
├── README.md
├── requirements.txt
└── Term Project - LSTM Model.py
```

The four output files listed above are generated when the script runs; they are not included in the repository by default.

## Disclaimer

For educational purposes only. This project is not financial advice or a trading tool.
