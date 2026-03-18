# API Example Requests

## POST /predict/booking

### curl
```bash
curl -X POST http://localhost:8000/predict/booking \
  -H "Content-Type: application/json" \
  -d '{
    "num_passengers": 1,
    "sales_channel": "Internet",
    "trip_type": "RoundTrip",
    "purchase_lead": 30,
    "length_of_stay": 7,
    "flight_hour": 10,
    "flight_day": "Mon",
    "route": "LHRBKK",
    "booking_origin": "United Kingdom",
    "wants_extra_baggage": 0,
    "wants_preferred_seat": 0,
    "wants_in_flight_meals": 0,
    "flight_duration": 11.5
  }'
```

### Python
```python
import requests

response = requests.post("http://localhost:8000/predict/booking", json={
    "num_passengers": 1,
    "sales_channel": "Internet",
    "trip_type": "RoundTrip",
    "purchase_lead": 30,
    "length_of_stay": 7,
    "flight_hour": 10,
    "flight_day": "Mon",
    "route": "LHRBKK",
    "booking_origin": "United Kingdom",
    "wants_extra_baggage": 0,
    "wants_preferred_seat": 0,
    "wants_in_flight_meals": 0,
    "flight_duration": 11.5,
})
print(response.json())
```

### Response
```json
{
  "booking_complete": 0,
  "probability": 0.7142,
  "model_used": "XGBoost"
}
```

---

## POST /sentiment/analyse

### curl
```bash
curl -X POST http://localhost:8000/sentiment/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "The crew was fantastic and the flight was very smooth."}'
```

### Python
```python
import requests

response = requests.post("http://localhost:8000/sentiment/analyse", json={
    "text": "The crew was fantastic and the flight was very smooth."
})
print(response.json())
```

### Response
```json
{
  "vader_neg": 0.0,
  "vader_neu": 0.423,
  "vader_pos": 0.577,
  "roberta_neg": 0.021,
  "roberta_neu": 0.134,
  "roberta_pos": 0.845,
  "avg_neg": 0.011,
  "avg_neu": 0.279,
  "avg_pos": 0.711,
  "pred_result": "Positive"
}
```

---

## GET /metrics/

### curl
```bash
curl http://localhost:8000/metrics/
```

### Response
```json
{
  "models": [
    {"model": "XGBoost",             "accuracy": 0.5683, "precision": 0.2336, "recall": 0.8284, "f1": 0.3645, "roc_auc": 0.6755},
    {"model": "Random Forest",       "accuracy": 0.6737, "precision": 0.2648, "recall": 0.6668, "f1": 0.3791, "roc_auc": 0.6708},
    {"model": "Logistic Regression", "accuracy": 0.5938, "precision": 0.2021, "recall": 0.5831, "f1": 0.3002, "roc_auc": 0.5894}
  ],
  "best_model": "XGBoost",
  "selected_features": ["length_of_stay", "booking_origin", "flight_duration"]
}
```

---

## GET /health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```
