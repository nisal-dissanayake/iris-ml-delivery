from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(iris.data, iris.target)
joblib.dump(model, Path(__file__).with_name("iris_model.joblib"))
print("Saved iris_model.joblib")
