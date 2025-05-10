from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    Generation,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatResult,
    LLMResult,
    BaseMessage,
    ChatGeneration
)
from typing import Any, Dict, List, Mapping, Optional, Union, Iterator
import requests
import json
from pydantic import Field, root_validator, model_validator
import time
from loguru import logger

from app.core.config import settings

class KhojLLM(LLM):
    """
    Integración de Khoj como LLM para LangChain.
    
    Khoj funciona como un servidor LLM local (autoalojado), enfocado en privacidad y eficiencia.
    """
    
    api_url: str = Field(default_factory=lambda: settings.KHOJ_API_URL)
    api_key: Optional[str] = Field(default_factory=lambda: settings.KHOJ_API_KEY)
    model: str = "default"  # Nombre del modelo en Khoj
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60  # Tiempo máximo de espera en segundos
    
    @property
    def _llm_type(self) -> str:
        return "khoj"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "api_url": self.api_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    @model_validator(mode='after')
    def validate_environment(self) -> "KhojLLM":
        """Validate that the API key is provided."""
        if not self.api_url:
            raise ValueError("Khoj API URL must be provided")
        return self
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Llamada al API de Khoj."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "query": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "model": self.model,
            }
            
            if stop:
                payload["stop"] = stop
                
            # Añadir cualquier parámetro adicional
            for key, value in kwargs.items():
                if key not in payload:
                    payload[key] = value
            
            # Llamada al API de Khoj
            response = requests.post(
                f"{self.api_url.rstrip('/')}/api/chat",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Procesar la respuesta
            result = response.json()
            if isinstance(result, dict):
                return result.get("response", "")
            else:
                return str(result)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al llamar a Khoj API: {str(e)}")
            return f"Error al conectar con Khoj: {str(e)}"
    
    def _generate(
        self, 
        prompts: List[str], 
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Genera resultados para múltiples prompts."""
        generations = []
        
        for prompt in prompts:
            generation_text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=generation_text)])
        
        return LLMResult(generations=generations)


class KhojChatModel(BaseChatModel):
    """
    Integración de Khoj como modelo de chat para LangChain.
    """
    
    api_url: str = Field(default_factory=lambda: settings.KHOJ_API_URL)
    api_key: Optional[str] = Field(default_factory=lambda: settings.KHOJ_API_KEY)
    model: str = "default"  # Nombre del modelo en Khoj
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60  # Tiempo máximo de espera en segundos
    streaming: bool = False
    
    @property
    def _llm_type(self) -> str:
        return "khoj_chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "api_url": self.api_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    @model_validator(mode='after')
    def validate_environment(self) -> "KhojChatModel":
        """Validate that the API key is provided."""
        if not self.api_url:
            raise ValueError("Khoj API URL must be provided")
        return self
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using Khoj."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Convertir mensajes a formato entendible por Khoj
            khoj_messages = []
            system_message = None
            
            for message in messages:
                if isinstance(message, SystemMessage):
                    system_message = message.content
                elif isinstance(message, HumanMessage):
                    khoj_messages.append({
                        "role": "user",
                        "content": message.content
                    })
                elif isinstance(message, AIMessage):
                    khoj_messages.append({
                        "role": "assistant",
                        "content": message.content
                    })
                else:
                    khoj_messages.append({
                        "role": "user",
                        "content": str(message.content)
                    })
            
            payload = {
                "messages": khoj_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "model": self.model,
            }
            
            if system_message:
                payload["system"] = system_message
                
            if stop:
                payload["stop"] = stop
                
            # Añadir cualquier parámetro adicional
            for key, value in kwargs.items():
                if key not in payload:
                    payload[key] = value
            
            # Llamada al API de Khoj
            response = requests.post(
                f"{self.api_url.rstrip('/')}/api/chat",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Procesar la respuesta
            result = response.json()
            message = AIMessage(content=result.get("response", ""))
            
            return ChatResult(generations=[ChatGeneration(message=message)])
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al llamar a Khoj API: {str(e)}")
            message = AIMessage(content=f"Error al conectar con Khoj: {str(e)}")
            return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _stream(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatResult]:
        """Implementación de streaming para Khoj."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Convertir mensajes a formato entendible por Khoj
            khoj_messages = []
            system_message = None
            
            for message in messages:
                if isinstance(message, SystemMessage):
                    system_message = message.content
                elif isinstance(message, HumanMessage):
                    khoj_messages.append({
                        "role": "user",
                        "content": message.content
                    })
                elif isinstance(message, AIMessage):
                    khoj_messages.append({
                        "role": "assistant",
                        "content": message.content
                    })
                else:
                    khoj_messages.append({
                        "role": "user",
                        "content": str(message.content)
                    })
            
            payload = {
                "messages": khoj_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "model": self.model,
                "stream": True
            }
            
            if system_message:
                payload["system"] = system_message
                
            if stop:
                payload["stop"] = stop
            
            # Si Khoj no soporta streaming, usar la versión no streaming
            if not self._check_streaming_support():
                result = self._generate(messages, stop, **kwargs)
                yield result
                return
            
            # Streaming con Khoj
            with requests.post(
                f"{self.api_url.rstrip('/')}/api/chat",
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                
                accumulated_message = ""
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8").lstrip("data: "))
                            if chunk.get("type") == "content":
                                content = chunk.get("content", "")
                                accumulated_message += content
                                message = AIMessage(content=accumulated_message)
                                yield ChatResult(generations=[ChatGeneration(message=message)])
                        except Exception as e:
                            logger.error(f"Error procesando chunk de streaming: {str(e)}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al llamar a Khoj API para streaming: {str(e)}")
            message = AIMessage(content=f"Error al conectar con Khoj: {str(e)}")
            yield ChatResult(generations=[ChatGeneration(message=message)])
    
    def _check_streaming_support(self) -> bool:
        """
        Verifica si el servidor Khoj soporta streaming.
        Fallback