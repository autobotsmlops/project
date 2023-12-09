import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import sys

#load dataset
def load_data(train_file_path,test_file_path):
    #loadin the dataset
    train = pd.read_csv(train_file_path)
    train = train.drop(columns=['Unnamed: 0','Timestamp'], axis=1)

    test = pd.read_csv(test_file_path)
    test = test.drop(columns=['Unnamed: 0','Timestamp'], axis=1)
    
    #dividing the data into train and test sets
    X_train = train.drop('Reading',axis=1)
    y_train = train['Reading']

    X_test = test.drop('Reading', axis=1)
    y_test = test['Reading']
    
    return X_train,X_test,y_train,y_test

def train(X_train,X_test,y_train,y_test): 
    # Start an MLflow run
    with mlflow.start_run() as run:

        # Train a Random Forest regressor
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)

        # Log model parameters and metrics using MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mse", mse)

        # Save the model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Get the model URI
        model_uri = f"runs:/{run.info.run_id}/random_forest_model"

    # Register the best model in the Model Registry
    model_version = mlflow.register_model(model_uri, "Predictive Maintenance Model")
    
    
def main():
    # python3 src/train.py data/prepared/train/train.csv data/prepared/test/test.csv
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    X_train,X_test,y_train,y_test = load_data(train_file_path,test_file_path)
    return

if __name__ == "__main__":
    main()
    