from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer
with open("../tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define Flask app
app = Flask(__name__)

# Preprocess input text
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)  # Adjust maxlen as per your training
    return padded_sequence

# Define API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get input JSON
    text = data.get("text", "")  # Extract text input
    
    if not text:
        return jsonify({"error": "No text provided"}), 400  # Error handling
    
    # Preprocess text & predict sentiment
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    
    # Convert prediction to human-readable output
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    return jsonify({"text": text, "sentiment": sentiment})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
