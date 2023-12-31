import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import yaml
import os
import joblib
from hyperopt import fmin, tpe, Trials
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe, space_eval
from hyperopt import hp
import pprint
import shutil

pp = pprint.PrettyPrinter(indent=4)

class Trainer:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        train = pd.read_csv(self.train_file_path)

        test = pd.read_csv(self.test_file_path)
        
        X_train = train.drop('Reading',axis=1)
        y_train = train['Reading']

        X_test = test.drop('Reading', axis=1)
        y_test = test['Reading']
        
        return X_train, X_test, y_train, y_test
    
    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def objective(self, params):
        model = RandomForestRegressor(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_test = self.y_test
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = self.mean_absolute_percentage_error(y_test, y_pred)
        
        return {'loss': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'status': STATUS_OK, 'model': model}

    def load_space(self):
        with open('src/params.yaml') as f:
            params = yaml.safe_load(f)
            param_grid = params['train_random_forest']
            
            # Define the search space
            space = hp.choice('random_forest', [
                {
                    'n_estimators': hp.choice('n_estimators', param_grid['n_estimators']),
                    'criterion': hp.choice('criterion', param_grid['criterion']),
                    'max_depth': hp.choice('max_depth', param_grid['max_depth']),
                    'min_samples_split': hp.choice('min_samples_split', param_grid['min_samples_split']),
                    'min_samples_leaf': hp.choice('min_samples_leaf', param_grid['min_samples_leaf']),
                    'max_features': hp.choice('max_features', param_grid['max_features']),
                    'random_state': hp.choice('random_state', param_grid['random_state']),
                }
            ])
            return space

    def train_grid(self):
        mlflow.end_run()
        mlflow.set_experiment("/best_experiment")    
        with mlflow.start_run(nested=True):
            trials = Trials()
            best = fmin(
                fn=self.objective,
                space=self.load_space(),
                algo=tpe.suggest,
                max_evals=8,
                trials=trials,
            )

            best_run = sorted(trials.results, key=lambda x: x['loss'])[0]

            mlflow.log_params(best)
            mlflow.log_metric("mse", best_run['loss'])
            mlflow.log_metric("rmse", best_run['rmse'])
            mlflow.log_metric("mae", best_run['mae'])
            mlflow.log_metric("r2", best_run['r2'])
            mlflow.log_metric("mape", best_run['mape'])
            mlflow.set_tag("best_model", mlflow.active_run().info.run_id)

            model = best_run['model']

            # Save the model with MLflow
            mlflow.sklearn.log_model(model, "best_model")
            # save and replace model
            model_path = "best_model"
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            mlflow.sklearn.save_model(model, model_path)
            
            print("Model trained and registered")
            print("Best run parameters:")
            pp.pprint(best)
            print("Best run metrics:")
            pp.pprint(best_run)

            mlflow.end_run()

    def train(self):
        mlflow.set_experiment("Random Forest Regression")
        self.train_grid()
        print("Done")

if __name__ == "__main__":
    trainer = Trainer('data/prepared/train/train.csv', 'data/prepared/test/test.csv')
    trainer.train()