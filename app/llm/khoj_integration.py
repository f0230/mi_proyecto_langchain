"""
Enhanced Khoj integration for LangChain with advanced features:
- Streaming support
- Rate limiting
- Error handling with fallbacks
- Memory management
- Content filtering
"""

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
from typing import Any, Dict, List, Mapping, Optional, Union, Iterator, Callable
import requests
import json
from pydantic import Field, model_validator
import time
import asyncio
from loguru import logger
import backoff
from functools import lru_cache

from app.core.config import settings

class KhojLLMEnhanced(LLM):
    """
    Enhanced Khoj integration as LLM for LangChain with additional features:
    - Rate limiting
    - Retry mechanism with backoff
    - Caching for repeated queries
    - Robust error handling
    """
    
    api_url: str = Field(default_factory=lambda: settings.KHOJ_API_URL)
    api_key: Optional[str] = Field(default_factory=lambda: settings.KHOJ_API_KEY)
    model: str = "default"  # Model name in Khoj
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60  # Maximum wait time in seconds
    retry_attempts: int = 3
    rate_limit_per_minute: int = 60
    enable_cache: bool = True
    fallback_message: str = "I'm having trouble connecting to the knowledge base. Let me provide a general response."
    
    # Internal state tracking
    _request_timestamps: List[float] = Field(default_factory=list)
    _cache_hits: int = 0
    _cache_misses: int = 0
    
    @property
    def _llm_type(self) -> str:
        return "khoj_enhanced"
    
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
    def validate_environment(self) -> "KhojLLMEnhanced":
        """Validate that the API URL is provided."""
        if not self.api_url:
            raise ValueError("Khoj API URL must be provided")
        return self
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        
        # Remove timestamps older than 60 seconds
        self._request_timestamps = [ts for ts in self._request_timestamps if now - ts < 60]
        
        # Check if we're over the rate limit
        if len(self._request_timestamps) >= self.rate_limit_per_minute:
            wait_time = 60 - (now - self._request_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Add current timestamp
        self._request_timestamps.append(time.time())
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        factor=2
    )
    def _make_api_call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Make API call to Khoj with retry and backoff."""
        self._enforce_rate_limit()
        
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
            
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Call Khoj API
        response = requests.post(
            f"{self.api_url.rstrip('/')}/api/chat",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        # Process response
        result = response.json()
        if isinstance(result, dict):
            return result.get("response", "")
        else:
            return str(result)
    
    @lru_cache(maxsize=100)
    def _cached_call(self, prompt_hash: str, prompt: str, stops_str: str, **kwargs) -> str:
        """Cached version of the API call."""
        # Convert stop list back to actual list
        stop = json.loads(stops_str) if stops_str else None
        return self._make_api_call(prompt, stop, **kwargs)
    
    def _get_prompt_hash(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Create a hash key for the prompt and parameters."""
        # Create a string representation of the stop list
        stop_str = json.dumps(stop) if stop else ""
        
        # Only include parameters that affect the output
        key_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model,
        }
        
        # Add any additional parameters that might affect output
        for key, value in kwargs.items():
            if key in ["top_p", "top_k", "presence_penalty", "frequency_penalty"]:
                key_params[key] = value
        
        # Create a string to hash
        hash_input = f"{prompt}::{stop_str}::{json.dumps(key_params, sort_keys=True)}"
        return str(hash(hash_input))
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Call to Khoj API with caching and rate limiting."""
        try:
            if self.enable_cache:
                # Create hash of prompt and parameters
                prompt_hash = self._get_prompt_hash(prompt, stop, **kwargs)
                # Convert stop list to string for hashing
                stop_str = json.dumps(stop) if stop else ""
                
                try:
                    # Try to use cached response
                    start_time = time.time()
                    response = self._cached_call(prompt_hash, prompt, stop_str, **kwargs)
                    elapsed = time.time() - start_time
                    self._cache_hits += 1
                    logger.debug(f"Cache hit: {prompt_hash} (retrieved in {elapsed:.3f}s)")
                    return response
                except Exception as e:
                    # If cache fails, make direct API call
                    self._cache_misses += 1
                    logger.warning(f"Cache error: {str(e)}. Making direct API call")
                    return self._make_api_call(prompt, stop, **kwargs)
            else:
                return self._make_api_call(prompt, stop, **kwargs)
                
        except Exception as e:
            logger.error(f"Error calling Khoj API: {str(e)}")
            return self.fallback_message
    
    def _generate(
        self, 
        prompts: List[str], 
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Generate results for multiple prompts."""
        generations = []
        
        for prompt in prompts:
            generation_text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=generation_text)])
        
        return LLMResult(generations=generations)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_calls = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_calls * 100) if total_calls > 0 else 0
        
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total_calls": total_calls,
            "hit_rate_percent": hit_rate
        }
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cached_call.cache_clear()
        logger.info("KhojLLM cache cleared")


class KhojChatModelEnhanced(BaseChatModel):
    """
    Enhanced Khoj integration as chat model for LangChain.
    """
    
    api_url: str = Field(default_factory=lambda: settings.KHOJ_API_URL)
    api_key: Optional[str] = Field(default_factory=lambda: settings.KHOJ_API_KEY)
    model: str = "default"  # Model name in Khoj
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60  # Maximum wait time in seconds
    streaming: bool = False
    retry_attempts: int = 3
    rate_limit_per_minute: int = 60
    enable_cache: bool = True
    fallback_message: str = "I'm having trouble connecting to the knowledge base. Let me provide a general response."
    
    # Internal state tracking
    _request_timestamps: List[float] = Field(default_factory=list)
    _cache_hits: int = 0
    _cache_misses: int = 0
    
    # Content filtering
    content_filter: Optional[Callable[[str], str]] = None
    
    @property
    def _llm_type(self) -> str:
        return "khoj_chat_enhanced"
    
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
    def validate_environment(self) -> "KhojChatModelEnhanced":
        """Validate that the API URL is provided."""
        if not self.api_url:
            raise ValueError("Khoj API URL must be provided")
        return self
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        
        # Remove timestamps older than 60 seconds
        self._request_timestamps = [ts for ts in self._request_timestamps if now - ts < 60]
        
        # Check if we're over the rate limit
        if len(self._request_timestamps) >= self.rate_limit_per_minute:
            wait_time = 60 - (now - self._request_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Add current timestamp
        self._request_timestamps.append(time.time())
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        factor=2
    )
    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Make API call to Khoj with retry and backoff."""
        self._enforce_rate_limit()
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model,
        }
        
        if system_message:
            payload["system"] = system_message
            
        if stop:
            payload["stop"] = stop
            
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Call Khoj API
        response = requests.post(
            f"{self.api_url.rstrip('/')}/api/chat",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        # Process response
        return response.json()
    
    @lru_cache(maxsize=100)
    def _cached_call(
        self, 
        messages_hash: str, 
        messages_json: str, 
        system_message: Optional[str] = None,
        stops_str: str = "", 
        **kwargs
    ) -> Dict[str, Any]:
        """Cached version of the API call."""
        # Convert parameters back from strings
        messages = json.loads(messages_json)
        stop = json.loads(stops_str) if stops_str else None
        
        return self._make_api_call(messages, system_message, stop, **kwargs)
    
    def _get_messages_hash(
        self, 
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Create a hash key for the messages and parameters."""
        # Convert messages to stable representation
        messages_str = json.dumps(messages, sort_keys=True)
        
        # Convert stop list to string
        stop_str = json.dumps(stop) if stop else ""
        
        # Only include parameters that affect the output
        key_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "system": system_message or ""
        }
        
        # Add any additional parameters that might affect output
        for key, value in kwargs.items():
            if key in ["top_p", "top_k", "presence_penalty", "frequency_penalty"]:
                key_params[key] = value
        
        # Create a string to hash
        hash_input = f"{messages_str}::{stop_str}::{json.dumps(key_params, sort_keys=True)}"
        return str(hash(hash_input))
    
    def _filter_content(self, content: str) -> str:
        """Apply content filtering if configured."""
        if self.content_filter and callable(self.content_filter):
            try:
                return self.content_filter(content)
            except Exception as e:
                logger.error(f"Error in content filter: {str(e)}")
                return content
        return content
    
    def _format_messages_for_khoj(self, messages: List[BaseMessage]) -> tuple:
        """Convert LangChain messages to Khoj format."""
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
        
        return khoj_messages, system_message
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using Khoj."""
        try:
            # Convert messages to Khoj format
            khoj_messages, system_message = self._format_messages_for_khoj(messages)
            
            if self.enable_cache:
                # Create hash of messages and parameters
                messages_hash = self._get_messages_hash(khoj_messages, system_message, stop, **kwargs)
                
                # Serialize data for caching
                messages_json = json.dumps(khoj_messages)
                stop_str = json.dumps(stop) if stop else ""
                
                try:
                    # Try to use cached response
                    start_time = time.time()
                    result = self._cached_call(
                        messages_hash, 
                        messages_json, 
                        system_message, 
                        stop_str, 
                        **kwargs
                    )
                    elapsed = time.time() - start_time
                    self._cache_hits += 1
                    logger.debug(f"Cache hit: {messages_hash} (retrieved in {elapsed:.3f}s)")
                except Exception as e:
                    # If cache fails, make direct API call
                    self._cache_misses += 1
                    logger.warning(f"Cache error: {str(e)}. Making direct API call")
                    result = self._make_api_call(khoj_messages, system_message, stop, **kwargs)
            else:
                result = self._make_api_call(khoj_messages, system_message, stop, **kwargs)
            
            # Apply content filtering if configured
            content = self._filter_content(result.get("response", ""))
            
            message = AIMessage(content=content)
            return ChatResult(generations=[ChatGeneration(message=message)])
        
        except Exception as e:
            logger.error(f"Error in Khoj chat generation: {str(e)}")
            message = AIMessage(content=self.fallback_message)
            return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _check_streaming_support(self) -> bool:
        """
        Check if Khoj server supports streaming.
        """
        try:
            # Make a simple HEAD request to check if streaming endpoint exists
            response = requests.head(
                f"{self.api_url.rstrip('/')}/api/chat",
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            # Check if server returns headers indicating streaming support
            return (
                response.status_code == 200 and
                "Transfer-Encoding" in response.headers and
                "chunked" in response.headers.get("Transfer-Encoding", "")
            )
        except Exception as e:
            logger.warning(f"Error checking streaming support: {str(e)}")
            return False
    
    def _stream(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatResult]:
        """Stream implementation for Khoj."""
        try:
            # Convert messages to Khoj format
            khoj_messages, system_message = self._format_messages_for_khoj(messages)
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
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
            
            # If Khoj doesn't support streaming, use the non-streaming version
            if not self._check_streaming_support():
                result = self._generate(messages, stop, **kwargs)
                yield result
                return
            
            # Stream with Khoj
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
                                
                                # Apply content filtering to accumulated message
                                filtered_content = self._filter_content(accumulated_message)
                                
                                message = AIMessage(content=filtered_content)
                                yield ChatResult(generations=[ChatGeneration(message=message)])
                        except Exception as e:
                            logger.error(f"Error processing streaming chunk: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error in Khoj streaming: {str(e)}")
            message = AIMessage(content=self.fallback_message)
            yield ChatResult(generations=[ChatGeneration(message=message)])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_calls = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_calls * 100) if total_calls > 0 else 0
        
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total_calls": total_calls,
            "hit_rate_percent": hit_rate
        }
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cached_call.cache_clear()
        logger.info("KhojChatModel cache cleared")
    
    async def agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of generate."""
        # Since Khoj API doesn't have async support yet, we run in a thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._generate(messages, stop, **kwargs))


# Content filtering examples

def basic_content_filter(content: str) -> str:
    """Basic content filter that removes certain patterns."""
    # Example: Remove potentially harmful instructions
    patterns = [
        r"(?i)how to hack",
        r"(?i)illegal activities",
        # Add more patterns as needed
    ]
    
    filtered_content = content
    for pattern in patterns:
        filtered_content = re.sub(pattern, "[FILTERED]", filtered_content)
    
    return filtered_content

def create_sensitive_info_filter(sensitive_patterns: List[str]) -> Callable[[str], str]:
    """Create a filter for specific sensitive information patterns."""
    def filter_fn(content: str) -> str:
        filtered_content = content
        for pattern in sensitive_patterns:
            filtered_content = re.sub(pattern, "[REDACTED]", filtered_content)
        return filtered_content
    
    return filter_fn