import os
from pydantic_settings import BaseSettings
from typing import Optional, Dict, List, Any, Union

class Settings(BaseSettings):
    # Configuración general
    PROJECT_NAME: str = "LangChain Khoj Integration"
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # Rutas de datos
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    
    # Configuración de LLM
    KHOJ_API_URL: str = "http://localhost:8000"  # URL por defecto de Khoj
    KHOJ_API_KEY: Optional[str] = None

    # OpenAI (alternativa a Khoj para embeddings o como LLM)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    # HuggingFace
    HF_TOKEN: Optional[str] = None
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vectorstore
    VECTORSTORE_TYPE: str = "chroma"  # Opciones: chroma, weaviate, pinecone, qdrant
    VECTORSTORE_DIR: str = os.path.join(DATA_DIR, "vectorstore")
    WEAVIATE_URL: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    
    # Configuración del procesamiento de documentos
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Email
    EMAIL_SERVER: Optional[str] = None
    EMAIL_USER: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    
    # SQL
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Asegurar que existan los directorios necesarios
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.VECTORSTORE_DIR, exist_ok=True)