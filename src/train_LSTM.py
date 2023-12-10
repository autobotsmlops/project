import sys
import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Load dataset
def load_data(train_file_path, test_file_path):
    train = pd.read_csv(train_file_path)
    train = train.drop(columns=['Unnamed: 0', 'Timestamp'], axis=1)

    test = pd.read_csv(test_file_path)
    test = test.drop(columns=['Unnamed: 0', 'Timestamp'], axis=1)

    # Divide the data into train and test sets
    X_train = train.drop('Reading', axis=1)
    y_train = train['Reading']

    X_test = test.drop('Reading', axis=1)
    y_test = test['Reading']

    return X_train, X_test, y_train, y_test

# LSTM model training
def train_lstm(X_train, X_test, y_train, y_test):
    # Scale the input features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape data for LSTM (samples, time steps, features)
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the LSTM model
    model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=2)

    # Make predictions on the test set
    y_pred = model.predict(X_test_reshaped)

    # Invert scaling for forecast
    inv_y_pred = np.concatenate((y_pred, X_test_scaled[:, 1:]), axis=1)
    inv_y_pred = scaler.inverse_transform(inv_y_pred)
    inv_y_pred = inv_y_pred[:, 0]

    # Invert scaling for actual
    y_test_reshaped = y_test.values.reshape((len(y_test), 1))
    inv_y_actual = np.concatenate((y_test_reshaped, X_test_scaled[:, 1:]), axis=1)
    inv_y_actual = scaler.inverse_transform(inv_y_actual)
    inv_y_actual = inv_y_actual[:, 0]

    # Evaluate the LSTM model
    mse = mean_squared_error(inv_y_actual, inv_y_pred)

    # Log model parameters and metrics using MLflow
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": 50,
            "batch_size": 32,
            "validation_split": 0.1
        })
        mlflow.log_metric("mse", mse)

        # Save the model with MLflow
        mlflow.keras.log_model(model, "lstm_model", signature=None, input_example=X_train)
        
        mlflow.sklearn.save_model(model, "lstm_model")

# Main function
def main():
    # Set the experiment name
    mlflow.set_experiment("LSTM")
    
    # python3 src/train.py data/prepared/train/train.csv data/prepared/test/test.csv
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    X_train, X_test, y_train, y_test = load_data(train_file_path, test_file_path)

    # Train LSTM model and log to MLflow
    train_lstm(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
