from fastapi import FastAPI

from api.routes.health import router as health_router
from api.routes.model import router as model_router
from api.routes.predict import router as predict_router

app = FastAPI()
app.include_router(health_router)
app.include_router(model_router)
app.include_router(predict_router)
