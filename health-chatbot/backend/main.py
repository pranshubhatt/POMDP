"""
FastAPI backend for the medical chatbot with enhanced POMDP implementation.
Uses keyword matching for symptom extraction (no external API dependencies).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import traceback
from medical_pomdp import MedicalPOMDP
import uvicorn

app = FastAPI(title="Medical Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store chat sessions
chat_sessions: Dict[str, MedicalPOMDP] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    message: str
    diseases: Dict[str, float]
    severity: float
    suggested_questions: List[str]
    precautions: List[str]

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"status": "API is running", "description": "Medical Chatbot API with POMDP"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return response."""
    try:
        # Get or create session
        if request.session_id not in chat_sessions:
            try:
                chat_sessions[request.session_id] = MedicalPOMDP()
            except ValueError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize chat session: {str(e)}"
                )
        
        pomdp = chat_sessions[request.session_id]
        
        # Extract symptoms from user message
        try:
            new_symptoms = await pomdp.extract_symptoms(request.message)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract symptoms: {str(e)}"
            )
        
        # Update belief state with new symptoms
        if new_symptoms:
            try:
                pomdp.update_belief_state(new_symptoms)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to update belief state: {str(e)}"
                )
        
        # Generate response
        try:
            response = pomdp.generate_response()
            return ChatResponse(**response)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        # Log the full traceback for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/session/{session_id}/state")
async def get_session_state(session_id: str):
    """Get the current state of a chat session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id].get_state_summary()

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a chat session and clean up resources."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 