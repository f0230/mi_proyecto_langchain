from langchain.embeddings import (
    OpenAIEmbeddings,
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings
)
from typing import Optional, Any, Dict
from loguru import logger
import os

from app.core.config import settings

def get_embeddings_model(model_name: Optional[str] = None, **kwargs):
    """
    Retorna un modelo de embeddings configurado.
    
    Args:
        model_name: Nombre específico del modelo a usar (opcional)
        **kwargs: Argumentos adicionales para el modelo de embeddings
        
    Returns:
        Modelo de embeddings configurado
    """
    # Usar modelo especificado o el de la configuración
    if model_name is None:
        model_name = settings.HF_EMBEDDING_MODEL
    
    # Determinar el tipo de modelo de embeddings a usar
    if settings.OPENAI_API_KEY:
        # Usar OpenAI si hay API key disponible
        logger.info(f"Usando OpenAI embeddings")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",  # Modelo más reciente
            openai_api_key=settings.OPENAI_API_KEY,
            **kwargs
        )
    elif "cohere" in model_name.lower() and getattr(settings, "COHERE_API_KEY", None):
        # Usar Cohere si se especifica modelo cohere y hay API key
        logger.info(f"Usando Cohere embeddings: {model_name}")
        return CohereEmbeddings(
            model=model_name,
            cohere_api_key=getattr(settings, "COHERE_API_KEY", None),
            **kwargs
        )
    else:
        # Por defecto usar HuggingFace/SentenceTransformers (local)
        logger.info(f"Usando embeddings locales con modelo: {model_name}")
        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error al cargar HuggingFaceEmbeddings: {str(e)}")
            logger.info("Usando SentenceTransformerEmbeddings como fallback")
            return SentenceTransformerEmbeddings(
                model_name=model_name,
                **kwargs
            )

def get_available_embedding_models():
    """
    Retorna una lista de modelos de embeddings disponibles.
    
    Returns:
        Lista de modelos disponibles con metadatos
    """
    available_models = []
    
    # Modelos OpenAI
    if settings.OPENAI_API_KEY:
        available_models.extend([
            {
                "name": "text-embedding-3-small",
                "provider": "OpenAI",
                "dimensions": 1536,
                "description": "Modelo de embeddings de OpenAI (text-embedding-3-small)",
                "requires_api": True
            },
            {
                "name": "text-embedding-3-large",
                "provider": "OpenAI",
                "dimensions": 3072,
                "description": "Modelo de embeddings de OpenAI de alta dimensionalidad (text-embedding-3-large)",
                "requires_api": True
            }
        ])
    
    # Modelos Cohere
    if getattr(settings, "COHERE_API_KEY", None):
        available_models.extend([
            {
                "name": "embed-multilingual-v3.0",
                "provider": "Cohere",
                "dimensions": 1024,
                "description": "Modelo multilingüe de Cohere (v3.0)",
                "requires_api": True
            },
            {
                "name": "embed-english-v3.0",
                "provider": "Cohere",
                "dimensions": 1024,
                "description": "Modelo en inglés de Cohere (v3.0)",
                "requires_api": True
            }
        ])
    
    # Modelos locales (HuggingFace/SentenceTransformers)
    available_models.extend([
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "provider": "HuggingFace",
            "dimensions": 384,
            "description": "Modelo ligero y rápido para embeddings (multilingüe)",
            "requires_api": False
        },
        {
            "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "provider": "HuggingFace",
            "dimensions": 384,
            "description": "Modelo multilingüe para embeddings de textos",
            "requires_api": False
        },
        {
            "name": "sentence-transformers/distiluse-base-multilingual-cased-v1",
            "provider": "HuggingFace",
            "dimensions": 512,
            "description": "Modelo destilado para múltiples idiomas",
            "requires_api": False
        }
    ])
    
    return available_models