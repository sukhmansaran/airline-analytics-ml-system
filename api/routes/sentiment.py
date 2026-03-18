"""Sentiment analysis endpoint."""

from fastapi import APIRouter, HTTPException
from api.schemas import SentimentInput, SentimentResult
from nlp.sentiment import _get_sia, roberta_scores

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


@router.post("/analyse", response_model=SentimentResult, summary="Analyse review sentiment")
def analyse_sentiment(data: SentimentInput) -> SentimentResult:
    """
    Analyse the sentiment of a British Airways review.

    Runs the text through both VADER (rule-based) and RoBERTa (transformer-based)
    models, averages their scores, and returns a combined Positive / Neutral / Negative
    classification alongside all raw scores.
    """
    try:
        sia = _get_sia()
        vader = sia.polarity_scores(data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VADER error: {e}")

    try:
        roberta = roberta_scores(data.text)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RoBERTa error: {e}. Model may still be downloading — try again shortly.",
        )

    avg_neg = (vader["neg"] + roberta["roberta_neg"]) / 2
    avg_neu = (vader["neu"] + roberta["roberta_neu"]) / 2
    avg_pos = (vader["pos"] + roberta["roberta_pos"]) / 2

    if avg_neg > avg_neu and avg_neg > avg_pos:
        result = "Negative"
    elif avg_neu > avg_neg and avg_neu > avg_pos:
        result = "Neutral"
    else:
        result = "Positive"

    return SentimentResult(
        vader_neg=round(vader["neg"], 4),
        vader_neu=round(vader["neu"], 4),
        vader_pos=round(vader["pos"], 4),
        roberta_neg=round(roberta["roberta_neg"], 4),
        roberta_neu=round(roberta["roberta_neu"], 4),
        roberta_pos=round(roberta["roberta_pos"], 4),
        avg_neg=round(avg_neg, 4),
        avg_neu=round(avg_neu, 4),
        avg_pos=round(avg_pos, 4),
        pred_result=result,
    )
