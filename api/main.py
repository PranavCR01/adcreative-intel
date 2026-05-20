import os

from fastapi import FastAPI

app = FastAPI(title="Creative Intelligence API")

ERROR_MESSAGES = {
    "model_not_loaded": "Model is warming up. Please retry in 30 seconds.",
    "invalid_image": "Could not process image. Please upload a JPG or PNG.",
    "inference_failed": "Scoring failed. Please try again.",
}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Slice 4: chat router
from api.routes.chat import router as chat_router
app.include_router(chat_router, prefix="/chat")

# Slice 5: upload router
from api.routes.upload import router as upload_router
app.include_router(upload_router, prefix="/upload")
