"""
Embedding Models for RAG Systems

This module provides embedding model wrappers for:
- SentenceTransformers (local)
- OpenAI embeddings (API)
- HuggingFace models (local)
"""

import os
from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embeddings: np.ndarray
    model: str
    dimension: int


class EmbeddingModel:
    """
    Unified interface for various embedding models.
    """
    
    SUPPORTED_MODELS = {
        # SentenceTransformers
        "all-MiniLM-L6-v2": {"type": "sentence_transformer", "dim": 384},
        "all-mpnet-base-v2": {"type": "sentence_transformer", "dim": 768},
        "bge-large-en-v1.5": {"type": "sentence_transformer", "dim": 1024},
        "e5-large-v2": {"type": "sentence_transformer", "dim": 1024},
        "gte-large": {"type": "sentence_transformer", "dim": 1024},
        
        # OpenAI
        "text-embedding-3-small": {"type": "openai", "dim": 1536},
        "text-embedding-3-large": {"type": "openai", "dim": 3072},
        "text-embedding-ada-002": {"type": "openai", "dim": 1536},
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the embedding model
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None
        self._client = None
        
        if model_name not in self.SUPPORTED_MODELS:
            print(f"Warning: {model_name} not in supported list, assuming sentence_transformer")
            self.model_type = "sentence_transformer"
            self.dimension = 768  # Default dimension
        else:
            model_info = self.SUPPORTED_MODELS[model_name]
            self.model_type = model_info["type"]
            self.dimension = model_info["dim"]
    
    def _load_model(self):
        """Lazy load the model."""
        if self.model_type == "sentence_transformer":
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise ImportError("Please install sentence-transformers")
                
                self._model = SentenceTransformer(self.model_name)
        
        elif self.model_type == "openai":
            if self._client is None:
                try:
                    from openai import OpenAI
                    from dotenv import load_dotenv
                except ImportError:
                    raise ImportError("Please install openai and python-dotenv")
                
                load_dotenv()
                self._client = OpenAI()
    
    def embed(self, texts: list[str] | str) -> EmbeddingResult:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            EmbeddingResult with embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        self._load_model()
        
        if self.model_type == "sentence_transformer":
            embeddings = self._embed_sentence_transformer(texts)
        elif self.model_type == "openai":
            embeddings = self._embed_openai(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        if self.normalize:
            embeddings = self._normalize(embeddings)
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model_name,
            dimension=self.dimension
        )
    
    def _embed_sentence_transformer(self, texts: list[str]) -> np.ndarray:
        """Embed using SentenceTransformers."""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100
        )
        return np.array(embeddings)
    
    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        """Embed using OpenAI API."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding (1D or 2D array)
            doc_embeddings: Document embeddings (2D array)
            
        Returns:
            Similarity scores array
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize if not already
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(query_norm, doc_norms.T).flatten()
        return similarities


def get_embeddings(
    texts: list[str] | str,
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True
) -> np.ndarray:
    """
    Convenience function to get embeddings.
    
    Args:
        texts: Text(s) to embed
        model_name: Embedding model name
        normalize: Whether to normalize embeddings
        
    Returns:
        Embeddings as numpy array
    """
    model = EmbeddingModel(model_name=model_name, normalize=normalize)
    result = model.embed(texts)
    return result.embeddings


class CachedEmbeddingModel(EmbeddingModel):
    """
    Embedding model with caching support.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize cached embedding model.
        
        Args:
            model_name: Name of the embedding model
            cache_dir: Directory to store embedding cache
            **kwargs: Additional arguments for EmbeddingModel
        """
        super().__init__(model_name=model_name, **kwargs)
        self.cache_dir = cache_dir or ".embedding_cache"
        self._cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if exists."""
        cache_file = os.path.join(self.cache_dir, f"{self.model_name}.npz")
        if os.path.exists(cache_file):
            try:
                data = np.load(cache_file, allow_pickle=True)
                self._cache = dict(data['cache'].item())
            except Exception:
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"{self.model_name}.npz")
        np.savez(cache_file, cache=self._cache)
    
    def embed(self, texts: list[str] | str) -> EmbeddingResult:
        """Embed with caching."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self._cache:
                results.append((i, self._cache[text_hash]))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_result = super().embed(texts_to_embed)
            
            for idx, (orig_idx, embedding) in enumerate(zip(indices_to_embed, new_result.embeddings)):
                text_hash = hash(texts[orig_idx])
                self._cache[text_hash] = embedding
                results.append((orig_idx, embedding))
            
            self._save_cache()
        
        # Sort by original index and extract embeddings
        results.sort(key=lambda x: x[0])
        embeddings = np.array([r[1] for r in results])
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model_name,
            dimension=self.dimension
        )


def list_available_models() -> dict:
    """List all available embedding models."""
    return EmbeddingModel.SUPPORTED_MODELS


if __name__ == "__main__":
    # Example usage
    print("Available embedding models:")
    for name, info in list_available_models().items():
        print(f"  {name}: dim={info['dim']}, type={info['type']}")
    
    print("\nTesting embedding generation:")
    texts = ["Hello world", "Machine learning is fascinating"]
    embeddings = get_embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

