"""
Ollama API Client for Local LLM Integration
Uses qwen2.5:7b model for text generation
"""
import httpx
import json
import asyncio
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 60.0
MAX_TOKENS = 1024


class OllamaClient:
    """Async client for Ollama API"""
    
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout)
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if Ollama server is running and model is available"""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                model_available = any(self.model in m for m in models)
                
                return {
                    "status": "healthy",
                    "ollama_running": True,
                    "model_available": model_available,
                    "configured_model": self.model,
                    "available_models": models
                }
            else:
                return {
                    "status": "error",
                    "ollama_running": True,
                    "error": f"Unexpected status code: {response.status_code}"
                }
        except httpx.ConnectError:
            return {
                "status": "error",
                "ollama_running": False,
                "error": "Cannot connect to Ollama. Is it running?"
            }
        except Exception as e:
            return {
                "status": "error",
                "ollama_running": False,
                "error": str(e)
            }
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama API
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            json_mode: If True, request JSON formatted response
            
        Returns:
            Dict with 'success', 'response', and optionally 'error'
        """
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or MAX_TOKENS
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if json_mode:
            payload["format"] = "json"
        
        try:
            response = await client.post("/api/generate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "model": data.get("model", self.model),
                    "done": data.get("done", True)
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "response": None
                }
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Request timed out",
                "response": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    async def generate_with_messages(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate using chat format with message history
        
        Args:
            messages: List of {'role': 'user'|'assistant'|'system', 'content': str}
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with 'success', 'response', and optionally 'error'
        """
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or MAX_TOKENS
            }
        }
        
        try:
            response = await client.post("/api/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("message", {})
                return {
                    "success": True,
                    "response": message.get("content", ""),
                    "model": data.get("model", self.model),
                    "done": data.get("done", True)
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "response": None
                }
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": "Request timed out",
                "response": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None
            }


# Global client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create global Ollama client"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


async def check_ollama_health() -> Dict[str, Any]:
    """Quick health check for Ollama"""
    client = get_ollama_client()
    return await client.health_check()
