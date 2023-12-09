# Use a base Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the Flask application code
COPY app/app.py .

# Expose the Flask application port
EXPOSE 5000

# Set the entrypoint command to run the Flask application
CMD ["python", "app.py"]