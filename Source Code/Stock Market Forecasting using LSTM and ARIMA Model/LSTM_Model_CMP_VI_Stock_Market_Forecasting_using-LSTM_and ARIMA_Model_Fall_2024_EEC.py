import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

sns.set_style("whitegrid")

def create_windowed_data(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(window_size):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

st.title("Stock Price Forecasting using LSTM Model")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    st.write("## Data Preview", df.head())

    data = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    window_size = 60
    train_size = int(len(data_scaled) * 0.8)
    test_size = len(data_scaled) - train_size
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size-window_size:]

    X_train, y_train = create_windowed_data(train_data, window_size)
    X_test, y_test = create_windowed_data(test_data, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    if "model" not in st.session_state:
        st.session_state.model = None
    if "history" not in st.session_state:
        st.session_state.history = None

    if st.button("Train Model"):
        with st.spinner("Training..."):
            st.session_state.model = build_lstm_model(window_size)
            # Increase patience to allow more epochs for convergence
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            # Increase max epochs to 200
            history = st.session_state.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=200, batch_size=32, verbose=1,
                callbacks=[early_stopping]
            )
            st.session_state.history = history.history
        st.success("Model Trained Successfully!")

    if st.session_state.history is not None:
        min_val_loss_epoch = np.argmin(st.session_state.history['val_loss']) + 1
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(st.session_state.history['loss'], label='Train Loss', color='blue')
        ax.plot(st.session_state.history['val_loss'], label='Validation Loss', color='red')
        ax.axvline(min_val_loss_epoch, linestyle='--', color='green', label=f'Convergence at Epoch {min_val_loss_epoch}')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs Validation Loss")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        st.pyplot(fig)
        st.write(f"### Model converged at epoch {min_val_loss_epoch}")

    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "y_test_unscaled" not in st.session_state:
        st.session_state.y_test_unscaled = None

    if st.button("Show Predictions") or "predictions" in st.session_state:
        if st.session_state.model is None:
            st.error("Please train the model first.")
        else:
            predictions = st.session_state.model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            st.session_state.predictions = predictions
            st.session_state.y_test_unscaled = y_test_unscaled

            test_dates = df.index[-len(y_test):]
            fig, ax = plt.subplots()
            ax.plot(test_dates, y_test_unscaled, label='Actual Prices', color='blue')
            ax.plot(test_dates, predictions, label='Predicted Prices', color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title("Stock Price Prediction")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Performance Metrics
            rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
            mape = mean_absolute_percentage_error(y_test_unscaled, predictions) * 100
            accuracy = 100 - mape

            st.write(f"### RMSE: {rmse:.4f}")
            st.write(f"### MAPE: {mape:.2f}%")
            st.write(f"### Prediction Accuracy: {accuracy:.2f}%")

            # Residual Analysis
            st.subheader("Residuals Analysis")
            residuals = y_test_unscaled - predictions
            residuals_flat = residuals.flatten()

            # Residuals over time
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(test_dates, residuals_flat, color='orange', alpha=0.6, label='Residuals')
            ax.axhline(y=0, color='black', linestyle='--')
            ax.set_title("Residuals Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Residuals")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Residual density plot
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.kdeplot(residuals_flat, fill=True, color='purple', ax=ax)
            ax.set_title("Density Plot of Residuals")
            ax.set_xlabel("Residuals")
            ax.set_ylabel("Density")
            st.pyplot(fig)

    if st.button("Show Future Forecast"):
        future_steps = 30
        future_input = data_scaled[-window_size:]
        future_predictions = []

        for _ in range(future_steps):
            future_input_reshaped = future_input.reshape((1, window_size, 1))
            next_prediction = st.session_state.model.predict(future_input_reshaped, verbose=0)[0, 0]
            future_predictions.append(next_prediction)
            future_input = np.append(future_input[1:], next_prediction).reshape(-1, 1)

        future_predictions_unscaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='B')[1:]

        fig, ax = plt.subplots()
        ax.plot(df.index, data['Close'], label='Historical Prices', color='blue')
        ax.plot(future_dates, future_predictions_unscaled, label='Forecasted Prices', color='green')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Future Stock Price Prediction")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        future_df = pd.DataFrame({"Date": future_dates, "Predicted": future_predictions_unscaled.flatten()})
        st.dataframe(future_df)