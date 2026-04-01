"""
app.py — SOS Classification API for Noor Al-Tariq
Hybrid approach: Rules → Local ML model (→ Gemini fallback later)

Deploy on Render.com (free tier) or any platform that runs Python.
"""

import os
import re
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from rules import rule_check, DEFAULT_PRIORITY

# ───────────────────────────────────────────────────
# Init
# ───────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow Flutter app to call this API

# Load the trained model (TF-IDF + LogisticRegression pipeline)
MODEL_PATH = os.environ.get("MODEL_PATH", "request_type_model.joblib")
type_model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")
print(f"   Classes: {list(type_model.classes_)}")

# Confidence threshold — below this, mark as low-confidence
CONF_THRESHOLD = 0.60


# ───────────────────────────────────────────────────
# Language detection (simple heuristic)
# ───────────────────────────────────────────────────
def detect_language(text: str) -> str:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    total = max(len(text), 1)
    if arabic_chars / total > 0.3:
        return "ar"
    return "en"


# ───────────────────────────────────────────────────
# Hybrid classify
# ───────────────────────────────────────────────────
def classify(text: str) -> dict:
    text = str(text).strip()
    if not text:
        return {"error": "Empty text"}

    language = detect_language(text)

    # 1) Check rules first (emergency, crowd, translation)
    rule_result = rule_check(text)
    if rule_result:
        return {
            "request_type": rule_result["request_type"],
            "priority": rule_result["priority"],
            "confidence": 1.0,
            "needs_ambulance": rule_result["needs_ambulance"],
            "language": language,
            "source": rule_result["source"],
            "short_summary": text[:200],
        }

    # 2) Local ML model
    proba = type_model.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    predicted_type = type_model.classes_[idx]
    confidence = float(proba[idx])

    # Assign default priority based on type
    priority = DEFAULT_PRIORITY.get(predicted_type, 3)

    # If confidence is low, flag it (Flutter can show a confirmation dialog)
    is_low_confidence = confidence < CONF_THRESHOLD

    return {
        "request_type": predicted_type,
        "priority": priority,
        "confidence": round(confidence, 3),
        "needs_ambulance": (predicted_type == "emergency_response" and priority == 5),
        "language": language,
        "source": "local_model",
        "low_confidence": is_low_confidence,
        "short_summary": text[:200],
    }


# ───────────────────────────────────────────────────
# API Routes
# ───────────────────────────────────────────────────
@app.route("/classify", methods=["POST"])
def classify_endpoint():
    """
    POST /classify
    Body: {"text": "أنا ضايعة في الحرم"}
    Returns: {"request_type": "navigation", "priority": 3, "confidence": 0.87, ...}
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with a 'text' field"}), 400

    result = classify(data["text"])
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """Simple health check for monitoring."""
    return jsonify({"status": "ok", "model_classes": list(type_model.classes_)})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Noor Al-Tariq SOS Classifier",
        "usage": "POST /classify with JSON body {'text': 'your SOS message'}",
        "health": "GET /health",
    })


# ───────────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
