import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent.agent import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    image_id: str
    message: str
    session_id: str = ""    # optional, default prevents 422
    vertical: str = "gaming"


class ChatResponse(BaseModel):
    answer: str
    trace: list
    image_id: str


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = await asyncio.to_thread(run_agent, request.image_id, request.message)
        return ChatResponse(**result)
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again.",
        )
