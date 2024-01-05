from flask import Flask, request, jsonify, render_template
import mlflow.sklearn
import pandas as pd
import sys
import mlflow
import os
import sys
from flask_cors import CORS

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project directory to the sys.path
project_dir = os.path.join(current_dir, "..")
sys.path.insert(0, project_dir)

from src.prepare_data import DataPreparer


def _predict(data):
    # Load the MLflow model
    model = mlflow.sklearn.load_model(f"best_model")

    # Use the model to make predictions on the input data
    prediction = model.predict(data)

    return prediction


def prepare_data(data):
    data_preparer = DataPreparer()

    return data_preparer.normalize_data(data)


def form_response(dict_request):
    try:
        data = pd.DataFrame([dict_request])
        data = prepare_data(data)
        response = _predict(data)
        print("response", response)
        response = response.tolist()
        return response
    except Exception as e:
        print(e)
        response = str(e)
        return response


# Create the Flask application
app = Flask(__name__)
cors = CORS(app)


# Define a route for the default URL, which loads the form
@app.route("/")
def form():
    return render_template("index.html")


# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        dict_request = request.get_json()

        print(dict_request)

        return {"status": 200, "message": None, "data": form_response(dict_request)}

    except Exception as e:
        print(e)
        error = str(e)
        return {"status": 400, "message": error, "data": None}


# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
