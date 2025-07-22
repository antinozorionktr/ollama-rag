import os
from dotenv import load_dotenv

# Force load .env file and override any existing env vars
load_dotenv(override=True)

class Config:
    # Ollama settings - force use smaller model
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Force override for OLLAMA_MODEL - check multiple sources
    model_from_env = os.getenv("OLLAMA_MODEL")
    print(f"🔍 OLLAMA_MODEL from environment: {model_from_env}")
    
    # Force use the smaller model regardless of what's in env
    OLLAMA_MODEL = "gemma3:1b"
    print(f"🔧 Config forced to use: OLLAMA_MODEL = {OLLAMA_MODEL}")
    
    # Vector store settings - IMPROVED for better accuracy
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # Smaller chunks for better precision
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # More overlap to preserve context
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # File upload settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    
    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Search settings - IMPROVED for better retrieval
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))  # Retrieve more chunks for comprehensive answers