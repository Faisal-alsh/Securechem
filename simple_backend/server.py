"""
Bare-bones secure backend for chemistry and biology research assistants
Single-file implementation with authentication and basic responses
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import uvicorn


# ============================================================================
# Configuration & Auth
# ============================================================================

CREDENTIALS = {
    "1122": {"role": "CHEM_RESEARCHER", "model": "chem-expert"},
    "3344": {"role": "BIO_RESEARCHER", "model": "bio-expert"}
}


# ============================================================================
# Models
# ============================================================================

class ChatRequest(BaseModel):
    researcher: str
    password: str
    chatting: str


class ChatResponse(BaseModel):
    answer: str


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Secure Research Assistant Backend")


def authenticate(password: str) -> dict:
    """Authenticate user and return their role info"""
    if password not in CREDENTIALS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    return CREDENTIALS[password]


def generate_response(query: str, role: str) -> str:
    """Generate response based on role and query"""
    # Mock responses - replace with actual model inference
    if role == "CHEM_RESEARCHER":
        return f"[Chemistry Expert] Regarding your question: '{query[:50]}...' - This would be answered by the chemistry model with RAG context from chemistry knowledge base."
    elif role == "BIO_RESEARCHER":
        return f"[Biology Expert] Regarding your question: '{query[:50]}...' - This would be answered by the biology model with RAG context from biology knowledge base."
    else:
        return "Unknown role"


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Secure Research Assistant API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/info")
async def info():
    """Get API information"""
    return {
        "available_roles": ["CHEM_RESEARCHER", "BIO_RESEARCHER"],
        "models": {
            "CHEM_RESEARCHER": "chem-expert",
            "BIO_RESEARCHER": "bio-expert"
        },
        "endpoints": ["/", "/health", "/info", "/chat"]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with authentication

    Args:
        request: ChatRequest with researcher name, password, and query

    Returns:
        ChatResponse with answer from appropriate model
    """
    # Authenticate
    user_info = authenticate(request.password)

    # Generate response based on role
    answer = generate_response(request.chatting, user_info["role"])

    return ChatResponse(answer=answer)


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Secure Research Assistant Backend")
    print("=" * 60)
    print("\n📋 Credentials:")
    print("  Chemistry: password=1122")
    print("  Biology:   password=3344")
    print("\n🚀 Starting server at http://localhost:8000")
    print("📖 API docs at http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
