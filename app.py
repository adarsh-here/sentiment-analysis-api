from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np

# Load model and tokenizer
model = tf.keras.models.load_model("model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Initialize Flask app
app = Flask(__name__)

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)

    prediction = model.predict(padded_sequence)
    sentiment = "Positive" if prediction >= 0.5 else "Negative"

    return jsonify({"text": text, "sentiment": sentiment})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
