from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    HTMLTextSplitter
)
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Union
import re
import os
from loguru import logger
from pathlib import Path

from app.core.config import settings
from app.processors.cleaner import TextCleaner

class DocumentProcessor:
    """Clase para procesar y dividir documentos en chunks."""
    
    @staticmethod
    def get_text_splitter(doc_type: str = "default"):
        """
        Retorna el text splitter apropiado según el tipo de documento.
        
        Args:
            doc_type: Tipo de documento ("default", "markdown", "python", "html")
        
        Returns:
            Un objeto text splitter de LangChain
        """
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
        if doc_type == "markdown":
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif doc_type == "python":
            return PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif doc_type == "html":
            return HTMLTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        else:
            # RecursiveCharacterTextSplitter es el más versátil
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
    
    @classmethod
    def process_documents(
        cls,
        documents: List[Dict[str, Any]],
        clean_text: bool = True,
        doc_type: str = "default",
        extract_metadata: bool = True
    ) -> List[Document]:
        """
        Procesa una lista de documentos:
        1. Limpia el texto si clean_text es True
        2. Divide en chunks
        3. Opcionalmente extrae metadatos adicionales
        
        Args:
            documents: Lista de documentos
            clean_text: Si es True, limpia el texto antes de dividirlo
            doc_type: Tipo de documento para seleccionar el splitter adecuado
            extract_metadata: Si es True, extrae metadatos adicionales
            
        Returns:
            Lista de objetos Document de LangChain
        """
        logger.info(f"Procesando {len(documents)} documentos")
        
        # Convertir a formato Document de LangChain si es necesario
        langchain_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                if "page_content" in doc and "metadata" in doc:
                    langchain_docs.append(Document(
                        page_content=doc["page_content"],
                        metadata=doc["metadata"]
                    ))
                else:
                    # Si es un diccionario pero sin la estructura correcta
                    content = doc.get("content", "") or doc.get("text", "") or str(doc)
                    metadata = doc.get("metadata", {}) or {}
                    langchain_docs.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
            elif isinstance(doc, Document):
                langchain_docs.append(doc)
            else:
                # Si es otro tipo, intentamos convertirlo a Document
                try:
                    content = str(doc)
                    langchain_docs.append(Document(
                        page_content=content,
                        metadata={}
                    ))
                except:
                    logger.warning(f"No se pudo convertir documento a formato Document: {type(doc)}")
        
        # Limpiar texto si es necesario
        if clean_text:
            for i, doc in enumerate(langchain_docs):
                langchain_docs[i].page_content = TextCleaner.clean_text(doc.page_content)
        
        # Dividir en chunks
        text_splitter = cls.get_text_splitter(doc_type)
        chunked_documents = text_splitter.split_documents(langchain_docs)
        
        logger.info(f"Se generaron {len(chunked_documents)} chunks a partir de {len(langchain_docs)} documentos")
        return chunked_documents
    
    @classmethod
    def process_file(
        cls, 
        file_path: str, 
        clean_text: bool = True,
        extract_metadata: bool = True
    ) -> List[Document]:
        """
        Procesa un archivo según su extensión.
        
        Args:
            file_path: Ruta al archivo
            clean_text: Si es True, limpia el texto
            extract_metadata: Si es True, extrae metadatos adicionales
            
        Returns:
            Lista de documentos procesados y divididos
        """
        from app.ingestors.document_loader import DocumentLoader
        
        file_extension = Path(file_path).suffix.lower()
        
        # Determinar el tipo de documento
        doc_type = "default"
        if file_extension in [".md", ".markdown"]:
            doc_type = "markdown"
        elif file_extension in [".py", ".ipynb"]:
            doc_type = "python"
        elif file_extension in [".html", ".htm"]:
            doc_type = "html"
        
        # Cargar el documento
        documents = DocumentLoader.load_file(file_path)
        
        # Procesar los documentos
        return cls.process_documents(
            documents, 
            clean_text=clean_text, 
            doc_type=doc_type,
            extract_metadata=extract_metadata
        )