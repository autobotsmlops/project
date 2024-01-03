# Use a base Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy Files - ignore the files in the .dockerignore file
COPY . .

# Expose the Flask application port
EXPOSE 5000

# Set the entrypoint command to run the Flask application and push the latest model to DVC
CMD dvc pull --force && python /app/app/app.py