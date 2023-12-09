import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yaml
import sys

def load_data(file_path):
    return pd.read_csv(file_path)

def normalize_data(df):
    scaler = preprocessing.MinMaxScaler()
    scaled_reading = scaler.fit_transform(df['Reading'].values.reshape(-1, 1))
    df['Reading'] = scaled_reading
    df = df.dropna() # Handle missing values (if any)
    df = df.drop(columns=['Machine_ID','Sensor_ID'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour_of_day'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['year'] = df['Timestamp'].dt.year
    return df

def split_data(df, split):
    features = df.drop('Reading', axis=1)
    label = df['Reading']
    features = pd.get_dummies(features)#one hot encoding categorical variables
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=split,random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df,test_df

def main():
    print("Normalizing the data")
    #load the path variables
    train_output_file = sys.argv[1]
    test_output_file = sys.argv[2]
    
    #load raw data
    #file_path = "data/raw/raw_sensor_data.csv"
    file_path = "data/raw/sensor_data.csv"
    df = load_data(file_path)
    
    #normalize reading
    df = normalize_data(df)
    
    #split data
    split=yaml.safe_load(open('src/params.yaml'))['prepare']['split']
    train_df, test_df = split_data(df,split)
    
    #writing the files to csv
    train_df.to_csv(train_output_file)
    test_df.to_csv(test_output_file)
    return


#runtime call:
# python3 src/prepare.py data/prepared/train/train.csv data/prepared/test.csv

if __name__ == "__main__":
    main()
    