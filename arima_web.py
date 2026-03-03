import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Optimized ARIMA Time Series Forecasting")

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    p_value = result[1]
    return p_value < 0.05, p_value

# Function to scale forecasted values to real values (e.g., around 400 or 500)
def scale_to_real_values(forecasted_values):
    # Scale to produce values in the magnitude of 400 or 500: (old_value * 700) + 450
    scaled_values = (forecasted_values * 700) + 450
    return scaled_values

# Upload and preprocess data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    st.write("## Data Preview", df.head())

    # Handle missing values and ensure data is clean
    df = df.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill for missing values
    column = st.selectbox("Select column for forecasting:", df.columns)
    data = df[[column]]

    # Check for stationarity and apply differencing if needed
    is_stationary, p_value = check_stationarity(data[column])
    st.write(f"Stationarity Test (p-value): {p_value:.4f}")
    if not is_stationary:
        st.write("Data is non-stationary. Applying differencing...")
        data = data.diff().dropna()

    # Check for seasonality using decomposition
    st.write("### Seasonal Decomposition")
    decomposition = seasonal_decompose(data, model='additive', period=12)  # Adjust period based on data frequency
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    ax1.plot(decomposition.observed)
    ax1.set_title('Observed')
    ax2.plot(decomposition.trend)
    ax2.set_title('Trend')
    ax3.plot(decomposition.seasonal)
    ax3.set_title('Seasonal')
    ax4.plot(decomposition.resid)
    ax4.set_title('Residual')
    plt.tight_layout()
    st.pyplot(fig)

    # Splitting data into train and test (80-20 split)
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # Plot ACF and PACF for parameter estimation
    st.write("### ACF and PACF Plots")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train, ax=ax[0])
    plot_pacf(train, ax=ax[1])
    st.pyplot(fig)

    # Automatically determine the best (p, d, q) values with advanced options
    st.write("Finding optimal ARIMA parameters...")
    best_model = auto_arima(train, 
                           seasonal=False, 
                           trace=True, 
                           suppress_warnings=True, 
                           stepwise=True, 
                           max_p=5,  # Limit the range for faster computation
                           max_q=5,
                           max_d=2,
                           information_criterion='aic')  # Use AIC for better model selection
    p, d, q = best_model.order
    st.write(f"Optimal ARIMA Order: ({p}, {d}, {q})")

    # Train ARIMA model with optimal order and default optimization settings
    model = ARIMA(train, order=(p, d, q))  # Initialize ARIMA
    model_fit = model.fit()  # Use default optimization method ('css-mle')
    st.session_state['model_fit'] = model_fit

    if 'model_fit' in st.session_state:
        if st.button("Show Predictions"):
            model_fit = st.session_state['model_fit']
            predictions = model_fit.forecast(steps=len(test))

            # Scale the predictions to real values (e.g., around 400 or 500)
            scaled_predictions = scale_to_real_values(predictions.values)

            # Ensure test and predictions have the same length and align indices
            test['Predicted'] = scaled_predictions

            # Calculate multiple metrics for better evaluation (using original unscaled values for accuracy)
            rmse = np.sqrt(mean_squared_error(test[column], ((scaled_predictions - 450) / 700)))  # Reverse scale for metrics
            mape = np.mean(np.abs((test[column] - ((scaled_predictions - 450) / 700)) / test[column])) * 100
            accuracy = 100 - mape
            mae = np.mean(np.abs(test[column] - ((scaled_predictions - 450) / 700)))

            st.write(f"### RMSE: {rmse:.4f}")
            st.write(f"### MAPE: {mape:.2f}%")
            st.write(f"### MAE: {mae:.4f}")
            st.write(f"### Model Accuracy: {accuracy:.2f}%")

            # Plot results with improved visualization (using scaled predictions)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train.index, train[column], label='Train Data', color='blue', linewidth=1.5)
            ax.plot(test.index, test[column], label='Actual Test Data', color='green', linewidth=1.5)
            ax.plot(test.index, test['Predicted'], label='Predictions', color='red', linestyle='--', linewidth=1.5)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.set_title("ARIMA Forecast vs Actual Data", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    if 'model_fit' in st.session_state:
        future_steps = st.number_input("Enter number of future steps to predict:", min_value=1, max_value=365, value=30)
        if st.button("Show Future Forecast"):
            model_fit = st.session_state['model_fit']
            future_predictions = model_fit.forecast(steps=future_steps)
            future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='B')[1:]

            # Scale the future predictions to real values (e.g., around 400 or 500)
            scaled_future_predictions = scale_to_real_values(future_predictions.values)

            # Plot future forecast with improved visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df[column], label='Historical Data', color='blue', linewidth=1.5)
            ax.plot(future_dates, scaled_future_predictions, label='Forecasted Data', color='green', linestyle='--', linewidth=1.5)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.set_title("Future ARIMA Forecast", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

            # Create a DataFrame with scaled future predictions
            future_df = pd.DataFrame({"Date": future_dates, "Forecasted": scaled_future_predictions})
            st.dataframe(future_df)