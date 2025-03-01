from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("disease_prediction_model.pkl")

# Define symptoms (Example)
symptom_list = ["fever", "cough", "fatigue", "headache"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get("symptoms", [])

    # Convert symptoms to model input format
    input_features = [1 if symptom in symptoms else 0 for symptom in symptom_list]

    # Predict disease
    prediction = model.predict([input_features])
    
    return jsonify({"disease": prediction[0]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)

