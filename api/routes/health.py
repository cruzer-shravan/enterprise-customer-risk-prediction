from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])

class HealthResponse(BaseModel):
    message: str

@router.get("/", response_model=HealthResponse)
def home():
    return {"message": "Churn Prediction API"}


@router.get("/health")
def health():
    return {"status": "ok"}
