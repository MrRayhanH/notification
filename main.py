from flask import Flask, request, jsonify
import joblib, pickle
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

clf = joblib.load("emoji_classifier.joblib")
label_map = pickle.load(open("label_map.pkl", "rb"))

_encoder = None
def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder

@app.post("/predict")
def predict():
    text = request.form.get("tittle")
    if not text:
        data = request.get_json(silent=True) or {}
        text = data.get("title") or data.get("tittle")
    if not text:
        return jsonify({"error": "title/tittle is required"}), 400

    encoder = get_encoder()
    emb = encoder.encode([text], normalize_embeddings=True)
    code = int(clf.predict(emb)[0])
    return jsonify({"label": label_map[code], "label_code": code})
