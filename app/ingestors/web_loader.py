from langchain.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain.document_loaders.playwright import Element
from typing import List, Dict, Any, Optional, Union
import requests
import httpx
from bs4 import BeautifulSoup
from loguru import logger
from urllib.parse import urlparse
import time
import json

class WebLoader:
    """Clase para cargar contenido desde sitios web."""
    
    @staticmethod
    def load_url(url: str, use_playwright: bool = False) -> List[Dict[str, Any]]:
        """
        Carga el contenido de una URL.
        
        Args:
            url: La URL a cargar
            use_playwright: Si True, usa Playwright para renderizar JavaScript
        
        Returns:
            Lista de documentos con el contenido de la página
        """
        logger.info(f"Cargando URL: {url} {'con Playwright' if use_playwright else 'con WebBaseLoader'}")
        
        try:
            if use_playwright:
                loader = PlaywrightURLLoader(
                    urls=[url],
                    remove_selectors=["header", "footer", "nav", ".ads", ".sidebar"],
                )
            else:
                loader = WebBaseLoader(url)
            
            return loader.load()
        except Exception as e:
            logger.error(f"Error al cargar URL {url}: {str(e)}")
            return []
    
    @staticmethod
    def load_urls(urls: List[str], use_playwright: bool = False) -> List[Dict[str, Any]]:
        """Carga múltiples URLs."""
        all_documents = []
        
        if use_playwright:
            try:
                loader = PlaywrightURLLoader(
                    urls=urls,
                    remove_selectors=["header", "footer", "nav", ".ads", ".sidebar"],
                )
                return loader.load()
            except Exception as e:
                logger.error(f"Error al cargar URLs con Playwright: {str(e)}")
        
        # Si no usamos Playwright o hubo un error, cargamos una por una
        for url in urls:
            documents = WebLoader.load_url(url, use_playwright=False)
            all_documents.extend(documents)
            time.sleep(1)  # Pausa para evitar sobrecarga del servidor
            
        return all_documents
    
    @staticmethod
    def scrape_with_bs4(url: str) -> Dict[str, Any]:
        """
        Scrape de una URL con BeautifulSoup para casos más específicos.
        
        Returns:
            Diccionario con el contenido extraído y metadatos
        """
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extraer información básica
            title = soup.title.string if soup.title else ""
            
            # Eliminar elementos no deseados
            for element in soup.select("script, style, header, footer, nav"):
                element.extract()
            
            # Extraer el texto principal
            main_content = soup.get_text(separator=" ", strip=True)
            
            # Extraer enlaces
            links = [a.get("href") for a in soup.find_all("a", href=True)]
            
            return {
                "page_content": main_content,
                "metadata": {
                    "source": url,
                    "title": title,
                    "links": links
                }
            }
        except Exception as e:
            logger.error(f"Error al hacer scraping de {url}: {str(e)}")
            return {
                "page_content": "",
                "metadata": {"source": url, "error": str(e)}
            }
    
    @staticmethod
    def fetch_api_data(
        url: str, 
        method: str = "GET", 
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict[str, Any], str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        async_req: bool = False
    ) -> Dict[str, Any]:
        """
        Obtiene datos de una API REST.
        
        Args:
            url: URL del endpoint
            method: Método HTTP (GET, POST, etc.)
            headers: Cabeceras HTTP
            params: Parámetros de consulta para la URL
            data: Datos para enviar en el cuerpo (form data o string)
            json_data: Datos JSON para enviar en el cuerpo
            async_req: Si es True, usa httpx de forma asíncrona
        
        Returns:
            Documento con el contenido y metadatos
        """
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        try:
            if async_req:
                with httpx.Client(timeout=30.0) as client:
                    response = client.request(
                        method, 
                        url, 
                        headers=headers,
                        params=params,
                        data=data,
                        json=json_data
                    )
                    response.raise_for_status()
            else:
                response = requests.request(
                    method, 
                    url, 
                    headers=headers,
                    params=params,
                    data=data,
                    json=json_data,
                    timeout=30
                )
                response.raise_for_status()
            
            # Intentar parsear como JSON
            try:
                content = response.json()
                page_content = json.dumps(content, ensure_ascii=False, indent=2)
            except:
                page_content = response.text
            
            return {
                "page_content": page_content,
                "metadata": {
                    "source": url,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": str(response.url)
                }
            }
        except Exception as e:
            logger.error(f"Error al acceder a la API {url}: {str(e)}")
            return {
                "page_content": "",
                "metadata": {"source": url, "error": str(e)}
            }