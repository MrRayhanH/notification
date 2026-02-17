from flask import Flask, request, jsonify
import os
import joblib, pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load model files
clf = joblib.load("emoji_classifier.joblib")
label_map = pickle.load(open("label_map.pkl", "rb"))

# Lazy-load encoder (slow on first request)
_encoder = None
def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


# ✅ Health / root endpoint (Render will stop giving 502 if health check passes)
@app.get("/")
def root():
    return jsonify({"status": "ok", "service": "emoji-notification-api"}), 200


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/predict")
def predict():
    # Accept JSON or form-data, and accept both 'title' and 'tittle'
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

    return jsonify({"label": label_map[code], "label_code": code}), 200


if __name__ == "__main__":
    # ✅ For local run only; Render uses gunicorn
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
