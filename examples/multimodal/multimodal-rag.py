"""
Multimodal RAG Implementation

Retrieval and reasoning across text, images, and audio
using vision-language models and multimodal embeddings.
"""

import os
import sys
import base64
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv
import numpy as np

from utils.loaders import Document, PDFLoader, ImageLoader, AudioLoader
from utils.chunking import RecursiveChunker
from utils.embeddings import EmbeddingModel
from utils.llm import get_llm
from utils.search import VectorSearch

load_dotenv()


class ModalityType(Enum):
    """Types of content modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TABLE = "table"


@dataclass
class MultimodalChunk:
    """Represents a chunk of multimodal content."""
    content: str
    modality: ModalityType
    metadata: dict = field(default_factory=dict)
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class MultimodalResult:
    """Result from multimodal RAG."""
    question: str
    answer: str
    retrieved_chunks: list[MultimodalChunk]
    modalities_used: list[str]
    image_descriptions: list[str]


class MultimodalRAG:
    """
    Multimodal RAG for text, images, and audio.
    
    Processes and retrieves from mixed-modality content
    using vision-language models and multimodal embeddings.
    """
    
    def __init__(
        self,
        vision_model: str = "gpt-4o",
        embedding_model: str = "all-MiniLM-L6-v2",
        audio_model: str = "whisper-base",
        chunk_size: int = 512,
        top_k: int = 5
    ):
        """
        Initialize Multimodal RAG.
        
        Args:
            vision_model: Model for image understanding
            embedding_model: Model for text embeddings
            audio_model: Model for audio transcription
            chunk_size: Size of text chunks
            top_k: Number of chunks to retrieve
        """
        self.vision_model = vision_model
        self.llm = get_llm(vision_model)
        self.embedding_model = EmbeddingModel(embedding_model)
        self.audio_model = audio_model
        self.top_k = top_k
        
        self.chunker = RecursiveChunker(chunk_size=chunk_size)
        self.vector_search = VectorSearch(backend="faiss")
        
        self.chunks: list[MultimodalChunk] = []
        self.indexed = False
    
    def _describe_image(self, image_path: str) -> str:
        """
        Generate description for an image using vision model.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Text description of the image
        """
        try:
            from openai import OpenAI
            client = OpenAI()
            
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            ext = os.path.splitext(image_path)[1].lower()
            media_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail, including any text, charts, diagrams, or data visible. Be specific about numbers and values."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Warning: Could not describe image: {e}")
            return f"[Image: {os.path.basename(image_path)}]"
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            loader = AudioLoader(model_size=self.audio_model.split('-')[-1])
            doc = loader.load(audio_path)
            return doc.content
        except Exception as e:
            print(f"Warning: Could not transcribe audio: {e}")
            return f"[Audio: {os.path.basename(audio_path)}]"
    
    def _extract_images_from_pdf(self, pdf_path: str) -> list[tuple[str, str]]:
        """
        Extract images from PDF and return paths with descriptions.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (image_path, description) tuples
        """
        try:
            import fitz
        except ImportError:
            print("Warning: pymupdf not installed, skipping image extraction")
            return []
        
        images = []
        doc = fitz.open(pdf_path)
        
        output_dir = os.path.join(os.path.dirname(pdf_path), ".images")
        os.makedirs(output_dir, exist_ok=True)
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha > 3:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    img_path = os.path.join(
                        output_dir,
                        f"page{page_num+1}_img{img_idx+1}.png"
                    )
                    pix.save(img_path)
                    
                    description = self._describe_image(img_path)
                    images.append((img_path, description))
                except Exception as e:
                    print(f"Warning: Could not extract image: {e}")
        
        doc.close()
        return images
    
    def index_text(self, text: str, metadata: dict = None):
        """
        Index text content.
        
        Args:
            text: Text content
            metadata: Optional metadata
        """
        metadata = metadata or {}
        text_chunks = self.chunker.chunk(text, metadata)
        
        for chunk in text_chunks:
            self.chunks.append(MultimodalChunk(
                content=chunk.content,
                modality=ModalityType.TEXT,
                metadata=chunk.metadata
            ))
    
    def index_image(self, image_path: str, metadata: dict = None):
        """
        Index an image by generating a description.
        
        Args:
            image_path: Path to image file
            metadata: Optional metadata
        """
        metadata = metadata or {}
        description = self._describe_image(image_path)
        
        self.chunks.append(MultimodalChunk(
            content=description,
            modality=ModalityType.IMAGE,
            metadata={**metadata, "image_path": image_path},
            image_path=image_path
        ))
    
    def index_audio(self, audio_path: str, metadata: dict = None):
        """
        Index audio by transcribing.
        
        Args:
            audio_path: Path to audio file
            metadata: Optional metadata
        """
        metadata = metadata or {}
        transcript = self._transcribe_audio(audio_path)
        
        # Chunk the transcript
        text_chunks = self.chunker.chunk(transcript, metadata)
        
        for chunk in text_chunks:
            self.chunks.append(MultimodalChunk(
                content=chunk.content,
                modality=ModalityType.AUDIO,
                metadata={**chunk.metadata, "audio_path": audio_path},
                audio_path=audio_path
            ))
    
    def index_pdf_with_images(self, pdf_path: str, metadata: dict = None):
        """
        Index a PDF including its images.
        
        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata
        """
        metadata = metadata or {}
        
        # Load text
        loader = PDFLoader()
        doc = loader.load(pdf_path)
        self.index_text(doc.content, {**metadata, "source": pdf_path})
        
        # Extract and index images
        images = self._extract_images_from_pdf(pdf_path)
        for img_path, description in images:
            self.chunks.append(MultimodalChunk(
                content=f"[Image from {os.path.basename(pdf_path)}]\n{description}",
                modality=ModalityType.IMAGE,
                metadata={**metadata, "source": pdf_path, "image_path": img_path},
                image_path=img_path
            ))
        
        print(f"Indexed PDF with {len(images)} images")
    
    def index_documents(self, documents: list):
        """
        Index multiple documents (text or dicts).
        
        Args:
            documents: List of documents
        """
        for doc in documents:
            if isinstance(doc, str):
                self.index_text(doc)
            elif isinstance(doc, Document):
                self.index_text(doc.content, doc.metadata)
            elif isinstance(doc, dict):
                self.index_text(doc["content"], doc.get("metadata", {}))
    
    def build_index(self):
        """Build the vector index from all chunks."""
        if not self.chunks:
            raise ValueError("No chunks to index")
        
        chunk_texts = [c.content for c in self.chunks]
        embeddings = self.embedding_model.embed(chunk_texts).embeddings
        
        for chunk, emb in zip(self.chunks, embeddings):
            chunk.embedding = emb
        
        chunk_docs = [
            {"content": c.content, "metadata": c.metadata}
            for c in self.chunks
        ]
        self.vector_search.index(chunk_docs, embeddings)
        
        self.indexed = True
        print(f"Built index with {len(self.chunks)} chunks")
        
        # Print modality distribution
        modality_counts = {}
        for chunk in self.chunks:
            modality_counts[chunk.modality.value] = modality_counts.get(chunk.modality.value, 0) + 1
        print(f"Modality distribution: {modality_counts}")
    
    def _answer_with_images(
        self,
        question: str,
        context: str,
        image_paths: list[str]
    ) -> str:
        """
        Generate answer using both text context and images.
        
        Args:
            question: User question
            context: Text context
            image_paths: Paths to relevant images
            
        Returns:
            Generated answer
        """
        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Build message with images
            content = [
                {
                    "type": "text",
                    "text": f"""Based on the following context and images, answer the question.

Context:
{context}

Question: {question}

Provide a comprehensive answer based on both the text and visual information."""
                }
            ]
            
            # Add images (limit to 3)
            for img_path in image_paths[:3]:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    ext = os.path.splitext(img_path)[1].lower()
                    media_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }.get(ext, 'image/jpeg')
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    })
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Warning: Falling back to text-only: {e}")
            return self.llm.generate(
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ).content
    
    def query(self, question: str) -> MultimodalResult:
        """
        Answer a question using multimodal retrieval.
        
        Args:
            question: The question to answer
            
        Returns:
            MultimodalResult with answer and retrieved content
        """
        if not self.indexed:
            self.build_index()
        
        # Retrieve relevant chunks
        query_embedding = self.embedding_model.embed(question).embeddings[0]
        results = self.vector_search.search(query_embedding, self.top_k)
        
        # Map results back to chunks
        retrieved_chunks = []
        for result in results.results:
            for chunk in self.chunks:
                if chunk.content == result.content:
                    retrieved_chunks.append(chunk)
                    break
        
        # Collect image paths and modalities
        image_paths = []
        modalities_used = set()
        image_descriptions = []
        
        for chunk in retrieved_chunks:
            modalities_used.add(chunk.modality.value)
            if chunk.image_path and os.path.exists(chunk.image_path):
                image_paths.append(chunk.image_path)
                image_descriptions.append(chunk.content)
        
        # Build context
        context = "\n\n---\n\n".join([
            f"[{chunk.modality.value.upper()}]\n{chunk.content}"
            for chunk in retrieved_chunks
        ])
        
        # Generate answer
        if image_paths:
            answer = self._answer_with_images(question, context, image_paths)
        else:
            answer = self.llm.generate(
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ).content
        
        return MultimodalResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            modalities_used=list(modalities_used),
            image_descriptions=image_descriptions
        )


def main():
    """Example usage of Multimodal RAG."""
    
    # Initialize
    print("🚀 Initializing Multimodal RAG...")
    rag = MultimodalRAG(
        vision_model="gpt-4o",
        embedding_model="all-MiniLM-L6-v2",
        top_k=5
    )
    
    # Sample documents with image references
    documents = [
        {
            "content": """
            Q4 2024 Financial Summary
            
            Total Revenue: $5.2 billion (up 15% YoY)
            
            Revenue by Segment:
            - Cloud Services: $2.1 billion (40%)
            - Enterprise Software: $1.8 billion (35%)
            - Consulting: $1.3 billion (25%)
            
            The accompanying chart shows quarterly revenue trends
            over the past year, demonstrating consistent growth.
            
            Key Highlights:
            - Cloud services grew 25% driven by AI offerings
            - Enterprise renewals at 95% rate
            - Consulting margins improved to 22%
            """,
            "metadata": {"source": "q4_report.pdf", "type": "financial"}
        },
        {
            "content": """
            Product Launch Announcement
            
            We are excited to announce the launch of AIStudio Pro,
            our next-generation AI development platform.
            
            Key Features:
            - No-code model training
            - One-click deployment
            - Built-in monitoring
            - Enterprise security
            
            Pricing starts at $99/month for individual developers.
            Enterprise plans available with custom pricing.
            
            See the product diagram for architecture overview.
            """,
            "metadata": {"source": "product_launch.pdf", "type": "announcement"}
        },
        {
            "content": """
            Customer Success Story: TechCorp Inc
            
            TechCorp implemented our AI platform to automate
            their customer service operations.
            
            Results after 6 months:
            - 40% reduction in response time
            - 25% cost savings
            - 95% customer satisfaction
            
            "The platform exceeded our expectations" 
            - Jane Smith, CTO of TechCorp
            
            Watch the video testimonial for the full story.
            """,
            "metadata": {"source": "case_study.pdf", "type": "case_study"}
        }
    ]
    
    # Index documents
    print("\n📚 Indexing documents...")
    rag.index_documents(documents)
    rag.build_index()
    
    # Test queries
    queries = [
        "What was the Q4 revenue and growth rate?",
        "What are the key features of AIStudio Pro?",
        "What results did TechCorp achieve?"
    ]
    
    for question in queries:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print("=" * 60)
        
        result = rag.query(question)
        
        print(f"\n🔍 Modalities used: {result.modalities_used}")
        print(f"📄 Chunks retrieved: {len(result.retrieved_chunks)}")
        
        if result.image_descriptions:
            print(f"🖼️ Images found: {len(result.image_descriptions)}")
        
        print("\n💡 Answer:")
        print("-" * 40)
        print(result.answer)


if __name__ == "__main__":
    main()

