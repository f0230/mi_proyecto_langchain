from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader, 
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from typing import List, Union, Dict, Any, Optional
import os
from loguru import logger
from pathlib import Path

from app.core.config import settings

class DocumentLoader:
    """Clase para cargar documentos de diferentes formatos."""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Dict[str, Any]]:
        """Carga un documento PDF."""
        logger.info(f"Cargando PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_docx(file_path: str) -> List[Dict[str, Any]]:
        """Carga un documento Word (.docx)."""
        logger.info(f"Cargando DOCX: {file_path}")
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_txt(file_path: str) -> List[Dict[str, Any]]:
        """Carga un archivo de texto plano."""
        logger.info(f"Cargando TXT: {file_path}")
        loader = TextLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_unstructured(file_path: str) -> List[Dict[str, Any]]:
        """Carga un documento utilizando UnstructuredFileLoader."""
        logger.info(f"Cargando con UnstructuredFileLoader: {file_path}")
        loader = UnstructuredFileLoader(file_path)
        return loader.load()
    
    @staticmethod
    def load_directory(dir_path: str, glob_pattern: str = "**/*.*") -> List[Dict[str, Any]]:
        """Carga todos los documentos de un directorio que coincidan con el patrón."""
        logger.info(f"Cargando directorio: {dir_path} con patrón: {glob_pattern}")
        loader = DirectoryLoader(
            dir_path, 
            glob=glob_pattern,
            loader_cls=UnstructuredFileLoader
        )
        return loader.load()
    
    @classmethod
    def load_file(cls, file_path: str) -> List[Dict[str, Any]]:
        """
        Detecta automáticamente el tipo de archivo y utiliza el cargador adecuado.
        """
        file_extension = Path(file_path).suffix.lower()
        
        if not os.path.exists(file_path):
            logger.error(f"El archivo no existe: {file_path}")
            return []

        try:
            if file_extension == ".pdf":
                return cls.load_pdf(file_path)
            elif file_extension in [".docx", ".doc"]:
                return cls.load_docx(file_path)
            elif file_extension == ".txt":
                return cls.load_txt(file_path)
            else:
                # Para otros formatos, intentamos con UnstructuredFileLoader
                return cls.load_unstructured(file_path)
        except Exception as e:
            logger.error(f"Error al cargar {file_path}: {str(e)}")
            return []

    @classmethod
    def load_files(cls, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Carga múltiples archivos y concatena sus resultados.
        """
        all_documents = []
        for file_path in file_paths:
            documents = cls.load_file(file_path)
            all_documents.extend(documents)
        return all_documents