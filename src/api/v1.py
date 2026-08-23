from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.fraud_detection.configs.config import settings
from src.fraud_detection.model.utils import load_model


app = FastAPI(
    title="Fraud Detection API"
)

# Load model once
model = load_model(settings.MODEL)


class PredictionRequest(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.post(
    "/predict",
    response_model=PredictResponse
)
async def predict(request: PredictionRequest):

    # Convert request to DataFrame
    features = pd.DataFrame([request.model_dump()])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    return PredictResponse(
        prediction=int(prediction),
        probability=float(probability)
    )


@app.get("/health")
async def health():
    return {"status": "good"}