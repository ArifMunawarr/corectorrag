"""LLM module using Ollama."""

import ollama
from typing import Optional, Dict, Any, List

from config import config


class OllamaLLM:
    """Wrapper for Ollama LLM."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.OLLAMA_MODEL
        self.client = ollama.Client(host=config.OLLAMA_BASE_URL)
        print(f"✓ LLM initialized: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_tokens: int = 256
    ) -> str:
        """Generate response from LLM."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        
        return response["message"]["content"]
    
    def check_connection(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            models = self.client.list()
            model_names = [m["name"] for m in models.get("models", [])]
            return any(self.model_name in name for name in model_names)
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return False


# Singleton instance
_llm = None


def get_llm() -> OllamaLLM:
    """Get or create singleton LLM instance."""
    global _llm
    if _llm is None:
        _llm = OllamaLLM()
    return _llm
