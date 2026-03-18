"""Model evaluation metrics endpoint."""

import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas import ModelMetrics, MetricsResponse

router = APIRouter(prefix="/metrics", tags=["Metrics"])

MODELS_DIR = os.getenv("MODELS_DIR", "models")


@router.get("/", response_model=MetricsResponse, summary="Get model evaluation metrics")
def get_metrics() -> MetricsResponse:
    """
    Return evaluation metrics for all trained models.

    Reads from models/scores.csv which is generated during training.
    Metrics include accuracy, precision, recall, F1, and ROC-AUC.
    """
    scores_path = os.path.join(MODELS_DIR, "scores.csv")
    meta_path = os.path.join(MODELS_DIR, "meta.json")

    if not os.path.exists(scores_path):
        raise HTTPException(status_code=404, detail="Metrics not found. Run training first.")

    df = pd.read_csv(scores_path)

    import json
    best_model = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        best_model = meta.get("best_model")

    models = [
        ModelMetrics(
            model=row["model"],
            accuracy=round(row["accuracy"], 4),
            precision=round(row["precision"], 4),
            recall=round(row["recall"], 4),
            f1=round(row["f1"], 4),
            roc_auc=round(row["roc_auc"], 4),
        )
        for _, row in df.iterrows()
    ]

    return MetricsResponse(
        models=models,
        best_model=best_model,
    )
