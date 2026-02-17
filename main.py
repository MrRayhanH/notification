from flask import Flask, request, jsonify
import os
import joblib, pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load model files
clf = joblib.load("emoji_classifier.joblib")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# Lazy-load encoder (slow on first request)
_encoder = None
def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder

# ✅ Root endpoint (Render health check)
@app.get("/")
def home():
    return "OK", 200

@app.get("/health")
def health():
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

    encoder = get_encoder()
    emb = encoder.encode([text], normalize_embeddings=True)
    code = int(clf.predict(emb)[0])

    # safer: label_map may have str keys depending how you saved it
    label = label_map.get(code) or label_map.get(str(code))

    return jsonify({"label": label, "label_code": code}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
