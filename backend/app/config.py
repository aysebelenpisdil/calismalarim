from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    PORT: int = 3001
    FRONTEND_URL: str = "http://localhost:3000"
    GEMINI_API_KEY: Optional[str] = None
    NODE_ENV: str = "development"
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # English-only, fast, 384 dimensions
    EMBEDDING_DIMENSION: int = 384
    
    # FAISS Index Configuration
    FAISS_INDEX_TYPE: str = "IndexFlatL2"  # Options: IndexFlatL2, IndexIVFFlat, IndexHNSW
    FAISS_METRIC: str = "L2"  # Options: L2 (Euclidean), IP (Inner Product)
    FAISS_INDEX_PATH: str = "data/recipe_index.faiss"
    
    # Reranker Configuration
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder for re-ranking
    RERANKER_BATCH_SIZE: int = 32  # Batch size for reranking
    RERANKER_ENABLED: bool = True  # Enable/disable reranker
    
    # LLM Configuration (Gemini)
    GEMINI_MODEL: str = "models/gemini-2.5-flash"  # Options: models/gemini-2.5-flash (fast), models/gemini-2.5-pro (quality), models/gemini-flash-latest
    GEMINI_MAX_TOKENS: int = 2000  # Maximum tokens for LLM response
    GEMINI_TEMPERATURE: float = 0.7  # Temperature for creativity (0.0-1.0)
    GEMINI_ENABLED: bool = True  # Enable/disable LLM explanations

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

