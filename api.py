from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    X = np.array([[
        data["gender"],
        data["age"],
        data["hypertension"],
        data["heart_disease"],
        data["smoking_history"],
        data["bmi"],
        data["HbA1c_level"],
        data["blood_glucose_level"]
    ]])

    prediction = model.predict(X)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
