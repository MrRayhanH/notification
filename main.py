from flask import Flask, request, jsonify
import os
import joblib, pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ✅ Absolute paths (works on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

clf = joblib.load(os.path.join(BASE_DIR, "emoji_classifier.joblib"))

with open(os.path.join(BASE_DIR, "label_map.pkl"), "rb") as f:
    label_map = pickle.load(f)

_encoder = None
def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder

@app.get("/")
def root():
    return jsonify({"status": "ok"}), 200

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (
        request.form.get("title")
        or request.form.get("tittle")
        or data.get("title")
        or data.get("tittle")
    )
    if not text:
        return jsonify({"error": "title (or tittle) is required"}), 400

    emb = get_encoder().encode([text], normalize_embeddings=True)
    code = int(clf.predict(emb)[0])
    return jsonify({"label": label_map[code], "label_code": code}), 200
