"""
Text Chunking Strategies for RAG Systems

This module provides various chunking strategies:
- RecursiveChunker: Hierarchical splitting with overlap
- SemanticChunker: Content-aware splitting based on embeddings
- SlidingWindowChunker: Fixed-size windows with overlap
"""

import re
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    metadata: dict
    index: int
    start_char: int
    end_char: int


class RecursiveChunker:
    """
    Recursively split text using a hierarchy of separators.
    
    Tries to split by paragraph first, then sentence, then word,
    keeping chunks close to the target size.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
        length_function: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order)
            length_function: Function to measure chunk length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n\n",  # Multiple blank lines (section break)
            "\n\n",    # Paragraph break
            "\n",      # Line break
            ". ",      # Sentence break
            ", ",      # Clause break
            " ",       # Word break
            ""         # Character break (last resort)
        ]
        self.length_function = length_function or len
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        chunks = self._split_text(text, self.separators)
        
        result = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunks):
            # Find position in original text
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            result.append(Chunk(
                content=chunk_text.strip(),
                metadata={
                    **metadata,
                    "chunk_size": len(chunk_text),
                    "chunking_method": "recursive"
                },
                index=i,
                start_char=start_pos,
                end_char=start_pos + len(chunk_text)
            ))
            
            current_pos = start_pos + len(chunk_text) - self.chunk_overlap
        
        return result
    
    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators."""
        if not text:
            return []
        
        # Check if text is already small enough
        if self.length_function(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator
        for separator in separators:
            if separator == "":
                # Last resort: split by characters
                return self._split_by_size(text)
            
            if separator in text:
                splits = text.split(separator)
                
                # Merge small splits and recursively process large ones
                merged = self._merge_splits(splits, separator)
                
                # Recursively process any still-too-large chunks
                result = []
                for chunk in merged:
                    if self.length_function(chunk) > self.chunk_size:
                        # Use remaining separators
                        remaining_separators = separators[separators.index(separator) + 1:]
                        result.extend(self._split_text(chunk, remaining_separators))
                    else:
                        result.append(chunk)
                
                return result
        
        return [text]
    
    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits together."""
        merged = []
        current = ""
        
        for split in splits:
            if not split.strip():
                continue
            
            test = current + separator + split if current else split
            
            if self.length_function(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    merged.append(current)
                current = split
        
        if current:
            merged.append(current)
        
        return merged
    
    def _split_by_size(self, text: str) -> list[str]:
        """Split text into fixed-size chunks (last resort)."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


class SemanticChunker:
    """
    Split text based on semantic similarity.
    
    Uses embeddings to find natural breakpoints where
    semantic content changes significantly.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """
        Initialize semantic chunker.
        
        Args:
            embedding_model: SentenceTransformer model name
            threshold: Similarity threshold for splitting
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._model = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Please install sentence-transformers")
            self._model = SentenceTransformer(self.embedding_model)
        return self._model
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of Chunk objects
        """
        import numpy as np
        
        metadata = metadata or {}
        model = self._load_model()
        
        # Split into sentences first
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(
                content=text.strip(),
                metadata={**metadata, "chunking_method": "semantic"},
                index=0,
                start_char=0,
                end_char=len(text)
            )]
        
        # Get embeddings for all sentences
        embeddings = model.encode(sentences)
        
        # Calculate similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(sim)
        
        # Find split points (where similarity drops below threshold)
        split_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                split_points.append(i + 1)
        split_points.append(len(sentences))
        
        # Create chunks from split points
        chunks = []
        current_pos = 0
        
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            
            chunk_text = " ".join(sentences[start_idx:end_idx])
            
            # Enforce size constraints
            if len(chunk_text) > self.max_chunk_size:
                # Split further if too large
                sub_chunks = self._split_by_size(chunk_text, self.max_chunk_size)
                for j, sub_chunk in enumerate(sub_chunks):
                    start_char = text.find(sub_chunk, current_pos)
                    if start_char == -1:
                        start_char = current_pos
                    
                    chunks.append(Chunk(
                        content=sub_chunk.strip(),
                        metadata={
                            **metadata,
                            "chunk_size": len(sub_chunk),
                            "chunking_method": "semantic"
                        },
                        index=len(chunks),
                        start_char=start_char,
                        end_char=start_char + len(sub_chunk)
                    ))
                    current_pos = start_char + len(sub_chunk)
            elif len(chunk_text) >= self.min_chunk_size:
                start_char = text.find(chunk_text, current_pos)
                if start_char == -1:
                    start_char = current_pos
                
                chunks.append(Chunk(
                    content=chunk_text.strip(),
                    metadata={
                        **metadata,
                        "chunk_size": len(chunk_text),
                        "chunking_method": "semantic"
                    },
                    index=len(chunks),
                    start_char=start_char,
                    end_char=start_char + len(chunk_text)
                ))
                current_pos = start_char + len(chunk_text)
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_size(self, text: str, max_size: int) -> list[str]:
        """Split text into chunks of maximum size."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        
        for word in words:
            if current_len + len(word) + 1 > max_size and current:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        
        if current:
            chunks.append(" ".join(current))
        
        return chunks


class SlidingWindowChunker:
    """
    Split text using a sliding window approach.
    
    Creates fixed-size chunks with configurable overlap,
    ensuring no context is lost between chunks.
    """
    
    def __init__(
        self,
        window_size: int = 512,
        stride: int = 256,
        respect_sentences: bool = True
    ):
        """
        Initialize sliding window chunker.
        
        Args:
            window_size: Size of each window in characters
            stride: Step size between windows
            respect_sentences: Try to end chunks at sentence boundaries
        """
        self.window_size = window_size
        self.stride = stride
        self.respect_sentences = respect_sentences
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Split text using sliding window.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        
        if len(text) <= self.window_size:
            return [Chunk(
                content=text.strip(),
                metadata={
                    **metadata,
                    "chunk_size": len(text),
                    "chunking_method": "sliding_window"
                },
                index=0,
                start_char=0,
                end_char=len(text)
            )]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.window_size
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:]
            else:
                chunk_text = text[start:end]
                
                # Respect sentence boundaries if enabled
                if self.respect_sentences:
                    # Find last sentence boundary
                    last_period = chunk_text.rfind('. ')
                    last_question = chunk_text.rfind('? ')
                    last_exclaim = chunk_text.rfind('! ')
                    
                    last_boundary = max(last_period, last_question, last_exclaim)
                    
                    if last_boundary > self.window_size // 2:
                        chunk_text = chunk_text[:last_boundary + 1]
                        end = start + last_boundary + 1
            
            chunks.append(Chunk(
                content=chunk_text.strip(),
                metadata={
                    **metadata,
                    "chunk_size": len(chunk_text),
                    "window_start": start,
                    "window_end": end,
                    "chunking_method": "sliding_window"
                },
                index=len(chunks),
                start_char=start,
                end_char=end
            ))
            
            start += self.stride
            
            # Stop if we've reached the end
            if start >= len(text):
                break
        
        return chunks


def create_chunker(
    method: str = "recursive",
    **kwargs
) -> RecursiveChunker | SemanticChunker | SlidingWindowChunker:
    """
    Factory function to create a chunker.
    
    Args:
        method: Chunking method (recursive, semantic, sliding_window)
        **kwargs: Additional arguments for the chunker
        
    Returns:
        Chunker instance
    """
    chunkers = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "sliding_window": SlidingWindowChunker
    }
    
    if method not in chunkers:
        raise ValueError(f"Unknown chunking method: {method}")
    
    return chunkers[method](**kwargs)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Artificial intelligence (AI) has transformed numerous industries. 
    Machine learning models can now process vast amounts of data.
    
    Natural language processing enables computers to understand human language.
    This has led to breakthroughs in translation and text generation.
    
    Computer vision allows machines to interpret visual information.
    Self-driving cars rely heavily on this technology.
    """
    
    print("Testing RecursiveChunker:")
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(sample_text)
    for chunk in chunks:
        print(f"  Chunk {chunk.index}: {len(chunk.content)} chars")
    
    print("\nText Chunking modules initialized successfully!")

