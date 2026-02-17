from flask import Flask, request, jsonify
import os
import joblib
import pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ✅ Always use absolute paths (works on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emoji_classifier.joblib")
LABEL_PATH = os.path.join(BASE_DIR, "label_map.pkl")

# ✅ Load model files once at startup
clf = joblib.load(MODEL_PATH)
with open(LABEL_PATH, "rb") as f:
    label_map = pickle.load(f)

# ✅ Lazy-load encoder (slow only on first request)
_encoder = None
def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder  # ✅ IMPORTANT: return must be outside the if


# ✅ Root / health endpoints (ONLY ONE "/" route)
@app.get("/")
def root():
    return jsonify({"status": "ok", "service": "emoji-notification-api"}), 200

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


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
    label = label_map.get(code, str(code)) if isinstance(label_map, dict) else label_map[code]

    return jsonify({"label": label, "label_code": code}), 200


if __name__ == "__main__":
    # ✅ Local run only; Render uses gunicorn
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
