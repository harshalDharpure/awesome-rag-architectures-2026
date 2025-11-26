"""
LLM Wrappers for RAG Systems

This module provides unified interfaces for:
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude 3)
- Local models (Llama 3, Mixtral via Ollama/vLLM)
- GGUF models (llama-cpp-python)
"""

import os
from typing import Optional, Generator
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: dict
    finish_reason: str


class BaseLLM(ABC):
    """Base class for LLM wrappers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response."""
        pass
    
    @abstractmethod
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream a response."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI GPT models wrapper."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model: Model name (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: OpenAI API key (uses env var if not provided)
        """
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason
        )
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream a response."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicLLM(BaseLLM):
    """Anthropic Claude models wrapper."""
    
    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        """
        Initialize Anthropic LLM.
        
        Args:
            model: Model name (claude-3-opus, claude-3-sonnet, claude-3-haiku)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: Anthropic API key
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature)
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason
        )
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream a response."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature)
        ) as stream:
            for text in stream.text_stream:
                yield text


class OllamaLLM(BaseLLM):
    """Ollama local models wrapper."""
    
    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model: Model name (llama3, mistral, mixtral, etc.)
            temperature: Sampling temperature
            base_url: Ollama server URL
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response."""
        import requests
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature)
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["response"],
            model=self.model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            },
            finish_reason="stop"
        )
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream a response."""
        import requests
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature)
                }
            },
            stream=True
        )
        response.raise_for_status()
        
        import json
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]


class LocalLLM(BaseLLM):
    """Local GGUF models via llama-cpp-python."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.7
    ):
        """
        Initialize local GGUF model.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: GPU layers (-1 for all)
            temperature: Sampling temperature
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("Please install llama-cpp-python")
        
        self.model_path = model_path
        self.temperature = temperature
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        output = self.llm(
            full_prompt,
            max_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", self.temperature),
            stop=["User:", "\n\nUser"]
        )
        
        return LLMResponse(
            content=output["choices"][0]["text"],
            model=os.path.basename(self.model_path),
            usage={
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "completion_tokens": output["usage"]["completion_tokens"],
                "total_tokens": output["usage"]["total_tokens"]
            },
            finish_reason=output["choices"][0]["finish_reason"]
        )
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream a response."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        stream = self.llm(
            full_prompt,
            max_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", self.temperature),
            stop=["User:", "\n\nUser"],
            stream=True
        )
        
        for output in stream:
            yield output["choices"][0]["text"]


class LLMWrapper:
    """
    Unified LLM interface that auto-selects the appropriate backend.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        Initialize LLM wrapper.
        
        Args:
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific arguments
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Auto-select backend based on model name
        if model.startswith("gpt-"):
            self._llm = OpenAILLM(model=model, temperature=temperature, max_tokens=max_tokens)
        elif model.startswith("claude-"):
            self._llm = AnthropicLLM(model=model, temperature=temperature, max_tokens=max_tokens)
        elif model.endswith(".gguf"):
            self._llm = LocalLLM(model_path=model, temperature=temperature, **kwargs)
        else:
            # Assume Ollama for other models
            self._llm = OllamaLLM(model=model, temperature=temperature, **kwargs)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response."""
        return self._llm.generate(prompt, system_prompt, **kwargs)
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream a response."""
        return self._llm.stream(prompt, system_prompt, **kwargs)
    
    def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Convenience method to generate and return content only."""
        response = self.generate(prompt, system_prompt, **kwargs)
        return response.content


def get_llm(
    model: str = "gpt-4o",
    **kwargs
) -> LLMWrapper:
    """
    Factory function to get an LLM wrapper.
    
    Args:
        model: Model identifier
        **kwargs: Additional arguments
        
    Returns:
        LLMWrapper instance
    """
    return LLMWrapper(model=model, **kwargs)


# RAG-specific prompt templates
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.

Guidelines:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Cite relevant parts of the context when possible
4. Be concise but comprehensive"""

RAG_USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


def create_rag_prompt(context: str, question: str) -> tuple[str, str]:
    """
    Create RAG prompts.
    
    Args:
        context: Retrieved context
        question: User question
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    return RAG_SYSTEM_PROMPT, user_prompt


if __name__ == "__main__":
    print("LLM Wrappers initialized successfully!")
    print("Available backends: OpenAI, Anthropic, Ollama, Local (GGUF)")

