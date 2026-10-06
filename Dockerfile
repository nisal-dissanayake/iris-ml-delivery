# Iris prediction service
# Build-time: install dependencies, copy source, TRAIN the model (iris_model.joblib is created inside the image layer).
# Run-time:   CMD starts the Flask API on port 5000.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so this layer is cached when only source changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Both Python files
COPY train.py app.py ./

# BUILD instruction: runs once during `docker build`, writes /app/iris_model.joblib into the image
RUN python train.py

# Documentation only: the app listens on 5000 (publishing is done with -p / compose ports)
EXPOSE 5000

# STARTUP configuration: runs every time a container starts
CMD ["python", "app.py"]
