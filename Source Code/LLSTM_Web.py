import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf

# Set Seaborn style
sns.set_style("whitegrid")

# Function to create windowed data
def create_windowed_data(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_lstm_model(window_size, lstm_units, dropout_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit App
st.title("Stock Price Forecasting using LSTM Model")

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Upload a CSV file with 'Date' and 'Close' columns.")
st.sidebar.write("2. Adjust the window size and hyperparameters.")
st.sidebar.write("3. Train the model and view predictions.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("CSV file must contain 'Date' and 'Close' columns.")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            st.write("## Data Preview", df.head())

            # Data preprocessing
            data = df[['Close']]
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data)

            # Dynamic window size selection
            window_size = st.slider("Select Window Size", min_value=10, max_value=100, value=60)

            # Split data into training and testing sets
            train_size = int(len(data_scaled) * 0.8)
            test_size = len(data_scaled) - train_size
            train_data, test_data = data_scaled[:train_size], data_scaled[train_size-window_size:]

            # Create windowed data
            X_train, y_train = create_windowed_data(train_data, window_size)
            X_test, y_test = create_windowed_data(test_data, window_size)

            # Reshape data for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Initialize session state for model and history
            if "model" not in st.session_state:
                st.session_state.model = None
            if "history" not in st.session_state:
                st.session_state.history = None

            # Hyperparameter tuning
            st.sidebar.subheader("Model Hyperparameters")
            lstm_units = st.sidebar.slider("Number of LSTM Units", min_value=10, max_value=200, value=50)
            dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2)
            batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32)

            # Train model
            if st.button("Train Model"):
                with st.spinner("Training..."):
                    st.session_state.model = build_lstm_model(window_size, lstm_units, dropout_rate)
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    epochs = 100
                    progress_bar = st.progress(0)
                    for epoch in range(epochs):
                        history = st.session_state.model.fit(
                            X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=1, batch_size=batch_size, verbose=0
                        )
                        progress_bar.progress((epoch + 1) / epochs)
                    st.session_state.history = history.history
                st.success("Model Trained Successfully!")

            # Plot training and validation loss
            if st.session_state.history is not None:
                min_val_loss_epoch = np.argmin(st.session_state.history['val_loss']) + 1
                fig, ax = plt.subplots()
                ax.plot(st.session_state.history['loss'], label='Train Loss', color='blue')
                ax.plot(st.session_state.history['val_loss'], label='Test Loss', color='red')
                ax.axvline(min_val_loss_epoch, linestyle='--', color='green', label=f'Convergence at Epoch {min_val_loss_epoch}')
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.set_title("Training vs Test Loss")
                ax.legend()
                st.pyplot(fig)
                st.write(f"### Model converged at epoch {min_val_loss_epoch}")

            # Make predictions
            if st.button("Show Predictions") or "predictions" in st.session_state:
                predictions = st.session_state.model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

                st.session_state.predictions = predictions
                st.session_state.y_test_unscaled = y_test_unscaled

                # Plot actual vs predicted prices
                test_dates = df.index[-len(y_test):]
                fig = px.line(title="Stock Price Prediction")
                fig.add_scatter(x=test_dates, y=y_test_unscaled.flatten(), name="Actual Prices", line=dict(color='blue'))
                fig.add_scatter(x=test_dates, y=predictions.flatten(), name="Predicted Prices", line=dict(color='red'))
                fig.update_layout(xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

                # Performance metrics
                rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
                mae = mean_absolute_error(y_test_unscaled, predictions)
                mape = mean_absolute_percentage_error(y_test_unscaled, predictions) * 100
                r2 = r2_score(y_test_unscaled, predictions)
                accuracy = 100 - mape

                st.write(f"### RMSE: {rmse:.4f}")
                st.write(f"### MAE: {mae:.4f}")
                st.write(f"### MAPE: {mape:.2f}%")
                st.write(f"### R² Score: {r2:.4f}")
                st.write(f"### Prediction Accuracy: {accuracy:.2f}%")

            # Future forecasting
            if st.button("Show Future Forecast"):
                future_steps = st.slider("Select Number of Future Steps", min_value=1, max_value=100, value=30)
                future_input = data_scaled[-window_size:]
                future_predictions = []

                for _ in range(future_steps):
                    future_input_reshaped = future_input.reshape((1, window_size, 1))
                    next_prediction = st.session_state.model.predict(future_input_reshaped, verbose=0)[0, 0]
                    future_predictions.append(next_prediction)
                    future_input = np.append(future_input[1:], next_prediction).reshape(-1, 1)

                future_predictions_unscaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='B')[1:]

                # Plot future predictions
                fig = px.line(title="Future Stock Price Prediction")
                fig.add_scatter(x=df.index, y=data['Close'], name="Historical Prices", line=dict(color='blue'))
                fig.add_scatter(x=future_dates, y=future_predictions_unscaled.flatten(), name="Forecasted Prices", line=dict(color='green'))
                fig.update_layout(xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

                # Display future predictions in a table
                future_df = pd.DataFrame({"Date": future_dates, "Predicted": future_predictions_unscaled.flatten()})
                st.dataframe(future_df)

    except Exception as e:
        st.error(f"Error reading file: {e}")