from fastapi import APIRouter, HTTPException

from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.services.model_service import model_service


router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        predictions = model_service.predict(request.records)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"predictions": predictions}
