from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import time
import logging
from app.config import settings
from app.routes import recipes, fridge
from app.services.faiss_service import faiss_service
from app.services.reranker_service import reranker_service
from app.services.llm_service import llm_service
from app.services.rag_pipeline import rag_pipeline

# Setup logger
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Fridge Chef API",
    description="Backend API for Smart Fridge Chef - AI-powered recipe recommendation system",
    version="1.0.0"
)

# Response time middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))  # ms
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],  # Frontend'in görebilmesi için
)


# Startup event - Initialize RAG Pipeline components
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler
    Initializes RAG Pipeline components:
    1. FAISS index (Retriever)
    2. Reranker service (lazy load)
    3. LLM service (lazy load)
    4. RAG Pipeline (coordinates all components)
    """
    logger.info("🚀 Starting Smart Fridge Chef API...")
    
    # Step 1: Load FAISS index (Retriever)
    try:
        logger.info("📦 Loading FAISS index (Retriever)...")
        success = faiss_service.load_index()
        
        if success:
            index_info = faiss_service.get_index_info()
            logger.info("✅ FAISS index loaded successfully")
            logger.info(f"   Index type: {index_info.get('index_type', 'unknown')}")
            logger.info(f"   Vectors: {index_info.get('num_vectors', 'unknown')}")
            logger.info(f"   Dimension: {index_info.get('dimension', 'unknown')}")
        else:
            logger.warning("⚠️  FAISS index not found or could not be loaded")
            logger.warning("   Vector search will not be available")
            logger.warning("   Application will continue with string matching fallback")
            
    except Exception as e:
        logger.error(f"❌ Error loading FAISS index: {e}", exc_info=True)
        logger.warning("   Continuing with string matching fallback")
    
    # Step 2: Initialize Reranker (lazy load - will load on first use)
    try:
        if reranker_service.enabled:
            logger.info("🔄 Reranker service initialized (will load on first use)")
            logger.info(f"   Model: {reranker_service.model_name}")
        else:
            logger.info("⚠️  Reranker is disabled in config")
    except Exception as e:
        logger.warning(f"⚠️  Reranker initialization warning: {e}")
    
    # Step 3: Initialize LLM service (lazy load - will load on first use)
    try:
        if llm_service.enabled:
            if llm_service.api_key:
                logger.info("🤖 LLM service initialized (will load on first use)")
                logger.info(f"   Model: {llm_service.model_name}")
            else:
                logger.warning("⚠️  GEMINI_API_KEY not found")
                logger.warning("   LLM explanations will not be available")
        else:
            logger.info("⚠️  LLM service is disabled in config")
    except Exception as e:
        logger.warning(f"⚠️  LLM initialization warning: {e}")
    
    # Step 4: RAG Pipeline is already initialized (singleton)
    logger.info("🔗 RAG Pipeline initialized")
    logger.info("   Components:")
    logger.info(f"      - Retriever (FAISS): {'✅' if faiss_service.is_loaded() else '❌'}")
    logger.info(f"      - Reranker: {'✅' if reranker_service.enabled else '❌'}")
    logger.info(f"      - Generator (LLM): {'✅' if (llm_service.enabled and llm_service.api_key) else '❌'}")
    
    logger.info("✅ API startup completed - RAG Pipeline ready")


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running
    Includes RAG Pipeline component status
    """
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.NODE_ENV,
        "rag_pipeline": {
            "retriever": {
                "available": faiss_service.is_loaded(),
                "type": "FAISS"
            },
            "reranker": {
                "available": reranker_service.enabled,
                "loaded": reranker_service.is_loaded() if reranker_service.enabled else False,
                "model": reranker_service.model_name if reranker_service.enabled else None
            },
            "generator": {
                "available": llm_service.is_available(),
                "model": llm_service.model_name if llm_service.enabled else None,
                "has_api_key": bool(llm_service.api_key) if llm_service.enabled else False
            }
        }
    }


# Include routers
app.include_router(recipes.router, prefix="/api")
app.include_router(fridge.router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Smart Fridge Chef API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True if settings.NODE_ENV == "development" else False
    )

