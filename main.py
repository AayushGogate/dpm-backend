from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# -------- LOAD & PREPARE DATA --------
df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
Y = df['Outcome']

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# -------- PREDICTION FUNCTION --------
def predict_diabetes(input_data):
    """
    input_data: list/tuple of 8 numeric values
    """
    arr = np.asarray(input_data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)
    return "Diabetic" if pred[0] == 1 else "Not Diabetic"

# -------- API ENDPOINT --------
@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    input_values = json_data.get("input")

    # Validate
    if input_values is None:
        return jsonify({"error": "No input received"}), 400

    if len(input_values) != 8:
        return jsonify({"error": "Expected 8 input values"}), 400

    result = predict_diabetes(input_values)
    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

