from fastapi import APIRouter

from api.services.model_service import model_service


router = APIRouter(tags=["model"])


@router.get("/model")
def model_metadata():
    return model_service.metadata()
