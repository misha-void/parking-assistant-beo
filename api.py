"""
FastAPI backend for the parking assistant.
Run with: uvicorn api:app --reload

Endpoints:
- POST /chat: Send a message and get a response
- GET /health: Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from parking_assistant.service import process_chat_message

app = FastAPI(
    title="Parking Assistant API",
    description="AI chatbot for parking lot information and reservations",
    version="0.1.0"
)


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's message", min_length=1)
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional chat history for context"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Assistant's response")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "parking-assistant"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message and return the assistant's response.
    
    Args:
        request: ChatRequest containing user message and optional history
    
    Returns:
        ChatResponse with the assistant's reply
    """
    try:
        # Convert Pydantic models to dicts for service layer
        history = None
        if request.history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.history]
        
        response = await process_chat_message(
            user_message=request.message,
            chat_history=history
        )
        
        return ChatResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


# Auto-generated interactive docs available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
