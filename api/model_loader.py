import json
import joblib
import os

MODELS_DIR = os.getenv("MODELS_DIR", "models")

_model = None
_scaler = None
_meta = None
_label_encoders = None


def load_artifacts():
    global _model, _scaler, _meta, _label_encoders
    model_path = os.path.join(MODELS_DIR, "booking_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, "meta.json")
    le_path = os.path.join(MODELS_DIR, "label_encoders.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training/train.py first.")

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    _label_encoders = joblib.load(le_path) if os.path.exists(le_path) else {}
    with open(meta_path) as f:
        _meta = json.load(f)


def get_model():
    if _model is None:
        load_artifacts()
    return _model


def get_scaler():
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_meta():
    if _meta is None:
        load_artifacts()
    return _meta


def get_label_encoders():
    if _label_encoders is None:
        load_artifacts()
    return _label_encoders


def is_loaded() -> bool:
    return _model is not None
