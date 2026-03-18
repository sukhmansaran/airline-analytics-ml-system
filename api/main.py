from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routes import predict, sentiment
from api.routes import metrics as metrics_router
from api.model_loader import load_artifacts, is_loaded
from api.schemas import HealthResponse
from nlp.sentiment import warm_up


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load booking model
    try:
        load_artifacts()
        print("Booking model artifacts loaded.")
    except FileNotFoundError:
        print("Warning: booking model not found. Run training/train.py first.")

    # Pre-load NLP models so first request doesn't time out
    print("Loading NLP models (VADER + RoBERTa) — this may take a moment on first run...")
    try:
        warm_up()
    except Exception as e:
        print(f"Warning: NLP model warm-up failed: {e}")

    yield


app = FastAPI(
    title="British Airways Data Science API",
    description="Sentiment analysis and booking prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict.router)
app.include_router(sentiment.router)
app.include_router(metrics_router.router)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=is_loaded())


@app.get("/")
def root():
    return {"message": "British Airways DS API. Visit /docs for Swagger UI."}
