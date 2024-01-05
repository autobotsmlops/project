import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yaml
import sys


class DataPreparer:
    def __init__(self, file_path=None):
        if file_path is not None:
            self.file_path = file_path
            self.df = self.load_data()

    def load_data(self):
        return pd.read_csv(self.file_path)

    def normalize_data(self, df=None):
        if df is None:
            df = self.df

        scaler = preprocessing.MinMaxScaler()
        # train data
        if "Reading" in df.columns:
            scaled_reading = scaler.fit_transform(df["Reading"].values.reshape(-1, 1))
            df["Reading"] = scaled_reading

        df = df.dropna()  # Handle missing values (if any)
        df = df.drop(columns=["Machine_ID", "Sensor_ID"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["second_of_minute"] = df["Timestamp"].dt.second
        df["minute_of_hour"] = df["Timestamp"].dt.minute
        df["hour_of_day"] = df["Timestamp"].dt.hour
        df = df.drop(columns=["Timestamp"])

        return df

    def split_data(self, split, df=None):
        if df is None:
            df = self.df

        features = df.drop("Reading", axis=1)
        label = df["Reading"]
        features = pd.get_dummies(features)  # one hot encoding categorical variables
        X_train, X_test, y_train, y_test = train_test_split(
            features, label, test_size=split, random_state=42
        )
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        return train_df, test_df


def main():
    print("Normalizing the data")
    # load the path variables
    train_output_file = sys.argv[1]
    test_output_file = sys.argv[2]

    # DataPrepare
    file_path = "data/raw/sensor_data.csv"
    data_preparer = DataPreparer(file_path)

    # normalize reading
    df = data_preparer.normalize_data()

    # split data
    split = yaml.safe_load(open("src/params.yaml"))["prepare"]["split"]
    train_df, test_df = data_preparer.split_data(split, df)

    # writing the files to csv
    train_df.to_csv(train_output_file)
    test_df.to_csv(test_output_file)
    return


# runtime call:
# python3 src/prepare.py data/prepared/train/train.csv data/prepared/test.csv

if __name__ == "__main__":
    main()
