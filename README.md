# Stock Market Analysis & Prediction

A machine learning web application for forecasting stock prices using **LSTM (Long Short-Term Memory)** and **ARIMA (AutoRegressive Integrated Moving Average)** models. Built with Streamlit, this project is focused on NEPSE (Nepal Stock Exchange) listed stocks.

---

## Overview

This project implements two time-series forecasting approaches to predict stock closing prices:

- **LSTM Model** — A deep learning model based on stacked LSTM layers, capable of learning long-term patterns in sequential stock data.
- **ARIMA Model** — A classical statistical model using auto-regression and moving averages, enhanced with stationarity checks (ADF test), seasonal decomposition, and auto-parameter tuning via `pmdarima`.

Both models are served through an interactive Streamlit web interface where users can upload their own CSV data, configure model parameters, train the model, and view forecasts with evaluation metrics.

---

## Features

- Upload any stock CSV file (OHLCV format)
- Automatic data preprocessing and scaling
- Train/test split with configurable ratios
- **LSTM Model**
  - Stacked LSTM layers with Dropout regularization
  - Sliding window-based input sequences
  - Early stopping to prevent overfitting
  - Future price forecasting (30 days ahead)
- **ARIMA Model**
  - Augmented Dickey-Fuller (ADF) stationarity test
  - Auto-differencing for non-stationary data
  - `auto_arima` for automatic order selection
  - ACF/PACF plots for manual diagnostics
  - Seasonal decomposition visualization
- Evaluation metrics: RMSE, R² Score, MAPE
- Interactive charts for historical vs. predicted prices

---

## Tech Stack

| Category        | Tools / Libraries                                  |
|-----------------|----------------------------------------------------|
| Language        | Python 3.x                                         |
| Web Framework   | Streamlit                                          |
| Deep Learning   | TensorFlow / Keras                                 |
| Statistical ML  | statsmodels, pmdarima                              |
| Data Processing | pandas, NumPy                                      |
| Visualization   | Matplotlib, Seaborn                                |
| Preprocessing   | scikit-learn (MinMaxScaler)                        |

---

## Project Structure

```
Stock Market Analysis & Prediction/
│
├── Source Code/
│   ├── LSTM_finaldefense.py          # LSTM Streamlit app
│   ├── ARIMA_finaldefense.py         # ARIMA Streamlit app
│   ├── arima_web.py                  # Optimized ARIMA app with auto_arima
│   ├── ARIMA_we.py                   # ARIMA variant
│   ├── LLSTM_Web.py                  # LSTM web version
│   ├── extract_nepse_data.py         # NEPSE data extraction script
│   │
│   ├── NABIL.csv                     # NABIL Bank stock data
│   ├── CHCL.csv                      # Chilime Hydropower stock data
│   ├── NFS.csv                       # NFS stock data
│   ├── SCB.csv                       # Standard Chartered Bank stock data
│   ├── stock_prices.csv              # Combined stock price dataset
│   │
│   ├── LSTM_finaldefense.ipynb       # LSTM development notebook
│   ├── ARIMA_finaldefense.ipynb      # ARIMA development notebook
│   ├── Stock_forecasting_Defense.ipynb
│   ├── final_version.ipynb
│   └── ...
│
├── Docs/
│   └── NABIL.csv
│
└── README.md
```

---

## Dataset

The project uses historical daily stock data from the **Nepal Stock Exchange (NEPSE)** in OHLCV format:

| Column | Description             |
|--------|-------------------------|
| Symbol | Stock ticker symbol     |
| Date   | Trading date            |
| Open   | Opening price           |
| High   | Daily high price        |
| Low    | Daily low price         |
| Close  | Closing price (target)  |
| Volume | Number of shares traded |

Sample stocks used: `NABIL` (NABIL Bank), `CHCL` (Chilime Hydropower), `NFS`, `SCB` (Standard Chartered Bank Nepal).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/chetanbabubhat9-sudo/Stock-market-Analysis-and-Prediction.git
cd Stock-market-Analysis-and-Prediction
```

### 2. Install dependencies

```bash
pip install streamlit tensorflow keras pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima
```

### 3. Run the LSTM app

```bash
streamlit run "Source Code/LSTM_finaldefense.py"
```

### 4. Run the ARIMA app

```bash
streamlit run "Source Code/ARIMA_finaldefense.py"
```

---

## How It Works

### LSTM Pipeline

1. Load and preprocess CSV data (parse dates, set index)
2. Normalize closing prices using `MinMaxScaler`
3. Generate sliding window sequences (window size = 60)
4. Split into 80% train / 20% test
5. Build stacked LSTM model (2 LSTM layers + Dropout + Dense)
6. Train with `EarlyStopping` callback
7. Predict on test set and inverse-transform results
8. Forecast next 30 days iteratively
9. Display metrics (RMSE, R², MAPE) and charts

### ARIMA Pipeline

1. Load and preprocess CSV data
2. Run ADF test to check stationarity
3. Apply differencing if non-stationary
4. Use `auto_arima` to find optimal (p, d, q) parameters
5. Fit ARIMA model on training data
6. Forecast on test set (last 30 days)
7. Plot ACF/PACF and seasonal decomposition
8. Display RMSE and prediction vs. actual chart

---

## Results

Both models are benchmarked on the same test split using:
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

The LSTM model captures non-linear long-term trends better, while ARIMA is faster and more interpretable for short-term forecasts.

---

## Authors

- Chetan Babu Bhat  
- Academic Project — EEC Fall 2024

---

## License

This project is for educational and research purposes.
