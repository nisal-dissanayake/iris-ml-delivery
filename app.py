from pathlib import Path

import joblib
from flask import Flask, jsonify, request


app = Flask(__name__)
model = joblib.load(Path(__file__).with_name("iris_model.joblib"))
names = ["setosa", "versicolor", "virginica"]


@app.get("/")
def home():
    return jsonify({"service": "iris-prediction", "usage": "v2: POST /predict with JSON {\"measurements\": [sepal_length, sepal_width, petal_length, petal_width]} in cm"})


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    measurements = body.get("measurements") if isinstance(body, dict) else None
    if (not isinstance(measurements, list) or len(measurements) != 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in measurements)):
        return jsonify({"error": "measurements must be four numbers"}), 400
    class_id = int(model.predict([measurements])[0])
    return jsonify({"prediction": names[class_id]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
