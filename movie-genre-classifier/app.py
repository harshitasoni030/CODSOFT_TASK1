"""Flask REST API + frontend for the movie genre classifier."""
import json
import logging
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "genre_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"
MIN_WORDS, MAX_CHARS = 10, 10000

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

model, meta = None, {}
try:
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
except FileNotFoundError:
    app.logger.warning("Model not found. Run `python train.py` first.")


def error(message, status):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok" if model else "model_missing"})


@app.get("/api/model-info")
def model_info():
    if model is None:
        return error("Model not trained yet. Run `python train.py`.", 503)
    return jsonify(meta)


@app.post("/api/predict")
def predict():
    if model is None:
        return error("Model not trained yet. Run `python train.py`.", 503)
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "plot" not in data:
        return error("Send JSON like {\"plot\": \"...\"}.", 400)
    plot = data["plot"]
    if not isinstance(plot, str) or not plot.strip():
        return error("The plot must be a non-empty string.", 400)
    plot = plot.strip()
    if len(plot) > MAX_CHARS:
        return error(f"The plot is too long (max {MAX_CHARS} characters).", 413)
    if len(plot.split()) < MIN_WORDS:
        return error(f"Write at least {MIN_WORDS} words so the model has enough to work with.", 422)

    try:
        genre = model.predict([plot])[0]
        result = {"genre": genre, "confidence": None, "top_predictions": [],
                  "model": meta.get("best_model", type(model[-1]).__name__)}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([plot])[0]
            ranked = sorted(zip(model.classes_, probs), key=lambda p: -p[1])[:5]
            result["top_predictions"] = [{"genre": g, "confidence": round(float(p), 4)} for g, p in ranked]
            result["confidence"] = result["top_predictions"][0]["confidence"]
        return jsonify(result)
    except Exception:
        app.logger.exception("Prediction failed")
        return error("Prediction failed unexpectedly. Please try again.", 500)


@app.errorhandler(404)
def not_found(_):
    return error("Not found.", 404)


@app.errorhandler(405)
def bad_method(_):
    return error("Method not allowed.", 405)


@app.errorhandler(500)
def server_error(_):
    return error("Internal server error.", 500)


if __name__ == "__main__":
    app.run(debug=True)
