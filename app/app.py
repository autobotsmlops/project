from flask import Flask, request, jsonify, render_template
import mlflow.sklearn
import pandas as pd
import sys
import mlflow
sys.path.insert(0, 'E:\School Projects\MLOPS\project') 
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

# Define a route for the default URL, which loads the form
@app.route("/")
def form():
    return render_template("index.html")

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        dict_request = request.get_json()
        
        return {
            "status": 200,
            "message": None,
            "data": form_response(dict_request)
        }

    except Exception as e:
        print(e)
        error = str(e)
        return {
            "status": 400,
            "message": error,
            "data": None
        }

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)