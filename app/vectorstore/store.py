from langchain.vectorstores import Chroma, Weaviate, Pinecone, Qdrant
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Union
import os
from loguru import logger

from app.core.config import settings
from app.vectorstore.embeddings import get_embeddings_model

class VectorStore:
    """Clase para interactuar con la base vectorial."""
    
    @staticmethod
    def get_vectorstore(
        documents: Optional[List[Document]] = None,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """
        Crea o carga una base vectorial según la configuración.
        
        Args:
            documents: Documentos a indexar (si es None, solo carga una base existente)
            collection_name: Nombre de la colección
            persist_directory: Directorio donde persistir la base (para Chroma)
            
        Returns:
            Instancia de la base vectorial
        """
        embeddings = get_embeddings_model()
        vectorstore_type = settings.VECTORSTORE_TYPE.lower()
        
        if persist_directory is None:
            persist_directory = os.path.join(settings.VECTORSTORE_DIR, collection_name)
        
        # Crear directorio si no existe
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        try:
            if vectorstore_type == "chroma":
                return VectorStore._get_chroma(embeddings, documents, persist_directory, collection_name)
            elif vectorstore_type == "weaviate":
                return VectorStore._get_weaviate(embeddings, documents, collection_name)
            elif vectorstore_type == "pinecone":
                return VectorStore._get_pinecone(embeddings, documents, collection_name)
            elif vectorstore_type == "qdrant":
                return VectorStore._get_qdrant(embeddings, documents, collection_name, persist_directory)
            else:
                logger.warning(f"Tipo de vectorstore {vectorstore_type} no reconocido. Usando Chroma por defecto.")
                return VectorStore._get_chroma(embeddings, documents, persist_directory, collection_name)
        except Exception as e:
            logger.error(f"Error al crear/cargar vectorstore: {str(e)}")
            # Fallback a Chroma local en caso de error
            logger.info("Fallback a Chroma local")
            return VectorStore._get_chroma(embeddings, documents, persist_directory, collection_name)
    
    @staticmethod
    def _get_chroma(
        embeddings,
        documents: Optional[List[Document]], 
        persist_directory: str,
        collection_name: str
    ):
        """Configura y retorna una instancia de Chroma."""
        if documents:
            # Crear nueva colección o añadir a existente
            logger.info(f"Creando/actualizando colección Chroma en {persist_directory}")
            return Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        else:
            # Cargar colección existente
            logger.info(f"Cargando colección Chroma desde {persist_directory}")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
    
    @staticmethod
    def _get_weaviate(
        embeddings,
        documents: Optional[List[Document]],
        collection_name: str
    ):
        """Configura y retorna una instancia de Weaviate."""
        try:
            import weaviate
            from weaviate.embedded import EmbeddedOptions
            
            # Si no hay URL configurada, usar Weaviate embedded
            if not settings.WEAVIATE_URL:
                client = weaviate.Client(
                    embedded_options=EmbeddedOptions(
                        persistence_data_path=os.path.join(settings.VECTORSTORE_DIR, "weaviate")
                    )
                )
            else:
                client = weaviate.Client(url=settings.WEAVIATE_URL)
            
            if documents:
                logger.info(f"Creando/actualizando colección Weaviate: {collection_name}")
                return Weaviate.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    client=client,
                    class_name=collection_name,
                    by_text=False
                )
            else:
                logger.info(f"Cargando colección Weaviate: {collection_name}")
                return Weaviate(
                    client=client,
                    index_name=collection_name,
                    text_key="text",
                    embedding=embeddings,
                    by_text=False
                )
        except ImportError:
            logger.error("Weaviate no está instalado. Fallback a Chroma.")
            return VectorStore._get_chroma(
                embeddings, 
                documents, 
                os.path.join(settings.VECTORSTORE_DIR, collection_name),
                collection_name
            )
        except Exception as e:
            logger.error(f"Error al configurar Weaviate: {str(e)}. Fallback a Chroma.")
            return VectorStore._get_chroma(
                embeddings, 
                documents, 
                os.path.join(settings.VECTORSTORE_DIR, collection_name),
                collection_name
            )
    
    @staticmethod
    def _get_pinecone(
        embeddings,
        documents: Optional[List[Document]],
        collection_name: str
    ):
        """Configura y retorna una instancia de Pinecone."""
        try:
            import pinecone
            
            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY no está configurado")
            
            # Inicializar Pinecone
            pinecone.init(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENVIRONMENT or "us-west1-gcp"
            )
            
            # Crear índice si no existe
            index_name = collection_name.lower()
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    metric="cosine",
                    dimension=1536  # Para OpenAI embeddings, ajustar según modelo
                )
            
            if documents:
                logger.info(f"Creando/actualizando índice Pinecone: {index_name}")
                return Pinecone.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    index_name=index_name
                )
            else:
                logger.info(f"Cargando índice Pinecone: {index_name}")
                return Pinecone.from_existing_index(
                    index_name=index_name,
                    embedding=embeddings
                )
        except (ImportError, ValueError) as e:
            logger.error(f"Error con Pinecone: {str(e)}. Fallback a Chroma.")
            return VectorStore._get_chroma(
                embeddings, 
                documents, 
                os.path.join(settings.VECTORSTORE_DIR, collection_name),
                collection_name
            )
    
    @staticmethod
    def _get_qdrant(
        embeddings,
        documents: Optional[List[Document]],
        collection_name: str,
        persist_directory: str
    ):
        """Configura y retorna una instancia de Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # Configurar cliente local
            client = QdrantClient(path=persist_directory)
            
            # Verificar si la colección existe
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            # Crear colección si no existe
            if collection_name not in collection_names:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # Para OpenAI embeddings, ajustar según modelo
                        distance=Distance.COSINE
                    )
                )
            
            if documents:
                logger.info(f"Creando/actualizando colección Qdrant: {collection_name}")
                return Qdrant.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=collection_name,
                    client=client
                )
            else:
                logger.info(f"Cargando colección Qdrant: {collection_name}")
                return Qdrant(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
        except ImportError:
            logger.error("Qdrant no está instalado. Fallback a Chroma.")
            return VectorStore._get_chroma(
                embeddings, 
                documents, 
                os.path.join(settings.VECTORSTORE_DIR, collection_name),
                collection_name
            )