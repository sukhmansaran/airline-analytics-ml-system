"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class BookingInput(BaseModel):
    num_passengers: int = Field(..., example=1)
    sales_channel: str = Field(..., example="Internet")
    trip_type: str = Field(..., example="RoundTrip")
    purchase_lead: int = Field(..., example=30)
    length_of_stay: int = Field(..., example=7)
    flight_hour: int = Field(..., example=10)
    flight_day: str = Field(..., example="Mon")
    route: str = Field(..., example="LHRBKK")
    booking_origin: str = Field(..., example="United Kingdom")
    wants_extra_baggage: int = Field(..., example=0)
    wants_preferred_seat: int = Field(..., example=0)
    wants_in_flight_meals: int = Field(..., example=0)
    flight_duration: float = Field(..., example=11.5)


class BookingPrediction(BaseModel):
    booking_complete: int
    probability: float
    model_used: str


class SentimentInput(BaseModel):
    text: str = Field(..., example="The flight was excellent and the crew was very helpful.")


class SentimentResult(BaseModel):
    vader_neg: float
    vader_neu: float
    vader_pos: float
    roberta_neg: float
    roberta_neu: float
    roberta_pos: float
    avg_neg: float
    avg_neu: float
    avg_pos: float
    pred_result: str


class ModelMetrics(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class MetricsResponse(BaseModel):
    models: list[ModelMetrics]
    best_model: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
