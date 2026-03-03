import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.title("ARIMA Time Series Forecasting")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    st.write("## Data Preview", df.head())

    # Selecting only the target column for ARIMA
    column = st.selectbox("Select column for forecasting:", df.columns)
    data = df[[column]]

    # Splitting data into train and test
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # Plot ACF and PACF
    st.write("### ACF and PACF Plots")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train, ax=ax[0])
    plot_pacf(train, ax=ax[1])
    st.pyplot(fig)

    # Train ARIMA model with fixed order (0,2,1)
    model = ARIMA(train, order=(0, 2, 1))
    model_fit = model.fit()
    st.session_state['model_fit'] = model_fit

    if 'model_fit' in st.session_state:
        if st.button("Show Predictions"):
            model_fit = st.session_state['model_fit']
            predictions = model_fit.forecast(steps=len(test))
            test['Predicted'] = predictions.values

            rmse = np.sqrt(mean_squared_error(test[column], test['Predicted']))
            mape = np.mean(np.abs((test[column] - test['Predicted']) / test[column])) * 100
            accuracy = 100 - mape

            st.write(f"### RMSE: {rmse:.4f}")
            st.write(f"### MAPE: {mape:.2f}%")
            st.write(f"### Model Accuracy: {accuracy:.2f}%")

            fig, ax = plt.subplots()
            ax.plot(train.index, train[column], label='Train Data', color='blue')
            ax.plot(test.index, test[column], label='Actual Test Data', color='green')
            ax.plot(test.index, test['Predicted'], label='Predictions', color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.set_title("ARIMA Forecast vs Actual Data")
            ax.legend()
            st.pyplot(fig)

    if 'model_fit' in st.session_state:
        future_steps = st.number_input("Enter number of future steps to predict:", min_value=1, max_value=365, value=30)
        if st.button("Show Future Forecast"):
            model_fit = st.session_state['model_fit']
            future_predictions = model_fit.forecast(steps=future_steps)
            future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='B')[1:]

            fig, ax = plt.subplots()
            ax.plot(df.index, df[column], label='Historical Data', color='blue')
            ax.plot(future_dates, future_predictions, label='Forecasted Data', color='green')
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.set_title("Future ARIMA Forecast")
            ax.legend()
            st.pyplot(fig)

            future_df = pd.DataFrame({"Date": future_dates, "Forecasted": future_predictions.values})
            st.dataframe(future_df)
