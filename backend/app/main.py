"""
Secure Backend for Isolated Research-Assistant LLMs
Main FastAPI application with authentication and RAG-enhanced LLM inference.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
from pathlib import Path

from .auth import AuthService, Credentials, AuthenticationError, AuthorizationError
from .rag_engine import RAGManager
from .model_loader import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Secure Research Assistant Backend",
    description="Isolated LLMs for Chemistry and Biology Research",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model"""
    researcher: str = Field(..., description="Researcher name")
    password: str = Field(..., description="Access password")
    chatting: str = Field(..., description="User query", min_length=1)


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str = Field(..., description="LLM response")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


# Global service instances
auth_service: Optional[AuthService] = None
rag_manager: Optional[RAGManager] = None
model_manager: Optional[ModelManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global auth_service, rag_manager, model_manager

    logger.info("Starting up Secure Research Assistant Backend...")

    # Initialize authentication service
    auth_service = AuthService()
    logger.info("Authentication service initialized")

    # Initialize RAG manager
    backend_path = Path(__file__).parent.parent
    data_path = backend_path / "data"
    rag_manager = RAGManager(str(data_path))
    logger.info(f"RAG manager initialized with data path: {data_path}")

    # Initialize model manager
    models_path = backend_path / "models"
    models_path.mkdir(exist_ok=True)

    # Use CPU for demo/testing if no GPU available
    model_manager = ModelManager(str(models_path), use_8bit=False)

    # Preload tokenizer
    model_manager.load_tokenizer()
    logger.info("Model manager initialized")

    logger.info("Startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    if model_manager:
        model_manager.cleanup()
    logger.info("Shutdown complete")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        message="Secure Research Assistant Backend is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="All services operational"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with authentication and RAG-enhanced LLM inference.

    Security:
    - Validates credentials
    - Enforces role-based access control
    - Isolates data access by domain (chem/bio)

    Process:
    1. Authenticate user and determine role
    2. Retrieve relevant context from domain-specific RAG database
    3. Generate response using appropriate LoRA adapter
    4. Return response

    Args:
        request: ChatRequest with researcher, password, and query

    Returns:
        ChatResponse with LLM-generated answer

    Raises:
        HTTPException: For authentication, authorization, or processing errors
    """
    try:
        # Step 1: Authenticate and authorize
        credentials = Credentials(
            researcher=request.researcher,
            password=request.password
        )

        role, model_adapter, rag_database = auth_service.validate_access(credentials)
        logger.info(f"User authenticated: role={role}, adapter={model_adapter}, rag={rag_database}")

        # Step 2: Retrieve relevant context from RAG database
        query = request.chatting
        context = rag_manager.get_context(rag_database, query)
        logger.info(f"Retrieved context from {rag_database} (length: {len(context)} chars)")

        # Step 3: Generate response using model with appropriate adapter
        answer = model_manager.generate_response(
            adapter_name=model_adapter,
            query=query,
            context=context,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

        logger.info(f"Generated response (length: {len(answer)} chars)")

        # Step 4: Return response
        return ChatResponse(answer=answer)

    except AuthenticationError as e:
        logger.warning(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

    except AuthorizationError as e:
        logger.warning(f"Authorization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/info")
async def get_info():
    """Get information about available models and roles"""
    return {
        "models": {
            "chem-expert": {
                "role": "CHEM_RESEARCHER",
                "description": "Chemistry research assistant with chemistry RAG database",
                "rag_database": "chem_rag"
            },
            "bio-expert": {
                "role": "BIO_RESEARCHER",
                "description": "Biology research assistant with biology RAG database",
                "rag_database": "bio_rag"
            }
        },
        "authentication": {
            "method": "password-based",
            "note": "Each role requires specific credentials"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
