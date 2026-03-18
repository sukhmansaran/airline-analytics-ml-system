"""Booking completion prediction endpoint."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas import BookingInput, BookingPrediction
from api.model_loader import get_model, get_scaler, get_meta, get_label_encoders
from training.feature_engineering import engineer_features

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/booking", response_model=BookingPrediction, summary="Predict booking completion")
def predict_booking(data: BookingInput) -> BookingPrediction:
    """
    Predict whether a customer will complete their flight booking.

    Applies the same label encoding and scaling used during training,
    then runs inference with the best-performing saved model.

    Returns the binary prediction, confidence probability, and model name.
    """
    try:
        model = get_model()
        scaler = get_scaler()
        meta = get_meta()
        label_encoders = get_label_encoders()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    all_features: list[str] = meta["all_features"]

    # Build a single-row DataFrame from input, apply the same feature engineering as training
    raw_row = pd.DataFrame([data.model_dump()])
    raw_row = engineer_features(raw_row)

    input_dict = raw_row.iloc[0].to_dict()
    row: dict = {}

    for col in all_features:
        val = input_dict.get(col, 0)
        if col in label_encoders:
            le = label_encoders[col]
            val = int(le.transform([val])[0]) if val in le.classes_ else 0
        row[col] = val

    try:
        input_df = pd.DataFrame([row], columns=all_features)
        scaled = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled, columns=all_features)
        pred = int(model.predict(scaled_df)[0])
        proba = float(model.predict_proba(scaled_df)[0][pred])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return BookingPrediction(
        booking_complete=pred,
        probability=round(proba, 4),
        model_used=meta["best_model"],
    )
