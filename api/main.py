import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="Creative Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Slice 4: chat router
from api.routes.chat import router as chat_router
app.include_router(chat_router, prefix="/chat")

# Slice 5: upload router
from api.routes.upload import router as upload_router
app.include_router(upload_router, prefix="/upload")
