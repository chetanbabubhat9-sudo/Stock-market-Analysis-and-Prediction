import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Stock Price Forecasting with ARIMA Model")

# File uploader for CSV
st.header("Upload Stock Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df[['Close']]

    # Display raw data
    st.subheader("Raw Stock Data")
    st.write(df)

    # Plot original data
    st.subheader("Stock Price Over Time")
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label="Historical Prices", color="blue")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(plt)

    # Split data into train and test
    data_train = data.iloc[:-30].copy()  # All but the last 30 rows (train)
    data_test = data.iloc[-30:].copy()   # Last 30 rows (test)

    # Check stationarity (ADF test)
    st.subheader("Stationarity Check (ADF Test)")
    adf_test = adfuller(data_train)
    st.write(f"ADF Test p-value: {adf_test[1]}")

    # Transform to stationary: differencing
    data_train_diff = data_train.diff().dropna()
    adf_test_diff = adfuller(data_train_diff)
    st.write(f"ADF Test p-value after differencing: {adf_test_diff[1]}")

    # Fit ARIMA model (using order (0,2,1) as in your notebook)
    st.subheader("ARIMA Model Fitting")
    model = ARIMA(data_train, order=(2, 0, 4))
    model_fit = model.fit()
    st.write("ARIMA Model Summary:")
    st.write(model_fit.summary())

    # Forecast for the next 30 days
    st.subheader("30-Day Stock Price Forecast")
    future_forecast = model_fit.forecast(30)

    # Create future dates
    future_dates = pd.date_range(start=data.index[-1], periods=31, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': future_dates, 'Close': future_forecast})
    forecast_df.set_index('Date', inplace=True)

    # Display the forecasted prices for the next 30 days in a table
    st.write("### Predicted Prices for the Next 30 Days")
    st.write(forecast_df)

    # Plot historical and forecasted prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label="Historical Prices", color="blue")
    plt.plot(forecast_df.index, forecast_df['Close'], label="Forecasted Prices (Next 30 Days)", color="red")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price Forecast for Next 30 Days")
    st.pyplot(plt)

    # Model evaluation on test set
    st.subheader("Model Evaluation")
    y_true = data_test['Close']
    y_pred = model_fit.get_forecast(steps=len(data_test)).predicted_mean

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mean_y_true = np.mean(y_true)
    accuracy_percent = (1 - (rmse / mean_y_true)) * 100

    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"Accuracy Percentage: {accuracy_percent:.2f}%")

    # Optional: Show residuals plot
    st.subheader("Residuals Analysis")
    residuals = model_fit.resid[1:]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(title='Density', kind='kde', ax=ax[1])
    plt.tight_layout()
    st.pyplot(plt)

    # Optional: ACF and PACF plots for residuals
    st.subheader("ACF and PACF of Residuals")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuals, ax=ax1)
    plot_pacf(residuals, ax=ax2)
    plt.tight_layout()
    st.pyplot(plt)
else:
    st.write("Please upload a CSV file containing stock data with 'Date' and 'Close' columns.")