from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import mlflow

# Specify the run ID of your MLflow run
run_id = "your-run-id"  # Replace with the actual run ID

# Load the MLflow model
model = mlflow.sklearn.load_model(f"runs:/{run_id}/random_forest_model")

# Get the path to the loaded model
model_path = mlflow.sklearn.get_model_path(model)

print("Model path:", model_path)

# Create the Flask application
app = Flask(__name__)

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Preprocess the input data
    df = pd.DataFrame(data)

    # Make predictions using the loaded model
    predictions = model.predict(df)

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)