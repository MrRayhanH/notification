# main.py
import os
import joblib, pickle
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

clf = joblib.load("emoji_classifier.joblib")
label_map = pickle.load(open("label_map.pkl", "rb"))
encoder = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def home():
    return "OK"

@app.post("/predict")
def predict():
    text = request.form.get("tittle")
    if text is None:
        data = request.get_json(silent=True) or {}
        text = data.get("title") or data.get("tittle")

    if not text:
        return jsonify({"error": "title/tittle is required"}), 400

    emb = encoder.encode([text], normalize_embeddings=True)
    code = int(clf.predict(emb)[0])
    label = label_map[code]
    return jsonify({"text": text, "label_code": code, "label": label})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
