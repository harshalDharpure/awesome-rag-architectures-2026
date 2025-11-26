"""
Document Loaders for RAG Systems

This module provides various document loaders for:
- PDF documents
- Web pages
- Images (with OCR)
- Audio files (with transcription)
"""

import os
import re
from typing import Optional
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class Document:
    """Represents a loaded document with content and metadata."""
    content: str
    metadata: dict
    source: str
    doc_type: str


class PDFLoader:
    """Load and extract text from PDF documents."""
    
    def __init__(self, extract_images: bool = False):
        """
        Initialize PDF loader.
        
        Args:
            extract_images: Whether to extract and OCR images from PDFs
        """
        self.extract_images = extract_images
    
    def load(self, file_path: str) -> Document:
        """
        Load a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document object with extracted content
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("Please install pymupdf: pip install pymupdf")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        doc = fitz.open(file_path)
        text_content = []
        
        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            text_content.append(f"[Page {page_num + 1}]\n{text}")
            
            # Optionally extract images and OCR them
            if self.extract_images:
                images = page.get_images()
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        img_text = self._ocr_image(pix)
                        if img_text:
                            text_content.append(f"[Image {img_index + 1} on Page {page_num + 1}]\n{img_text}")
                    except Exception:
                        continue
        
        doc.close()
        
        return Document(
            content="\n\n".join(text_content),
            metadata={
                "pages": len(doc),
                "file_name": os.path.basename(file_path)
            },
            source=file_path,
            doc_type="pdf"
        )
    
    def _ocr_image(self, pix) -> Optional[str]:
        """Run OCR on an image pixmap."""
        try:
            import pytesseract
            from PIL import Image
            import io
            
            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Run OCR
            text = pytesseract.image_to_string(img)
            return text.strip() if text.strip() else None
        except ImportError:
            return None
        except Exception:
            return None
    
    def load_directory(self, dir_path: str) -> list[Document]:
        """Load all PDFs from a directory."""
        documents = []
        for filename in os.listdir(dir_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(dir_path, filename)
                documents.append(self.load(file_path))
        return documents


class WebLoader:
    """Load and extract text from web pages."""
    
    def __init__(
        self,
        timeout: int = 30,
        user_agent: str = "Mozilla/5.0 (compatible; RAGBot/1.0)"
    ):
        """
        Initialize web loader.
        
        Args:
            timeout: Request timeout in seconds
            user_agent: User agent string for requests
        """
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}
    
    def load(self, url: str) -> Document:
        """
        Load a web page and extract text content.
        
        Args:
            url: URL of the web page
            
        Returns:
            Document object with extracted content
        """
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Get title
        title = soup.title.string if soup.title else ""
        
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            # Extract text and clean it
            text = main_content.get_text(separator='\n')
            text = self._clean_text(text)
        else:
            text = ""
        
        return Document(
            content=text,
            metadata={
                "title": title,
                "url": url
            },
            source=url,
            doc_type="webpage"
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()
    
    def load_multiple(self, urls: list[str]) -> list[Document]:
        """Load multiple web pages."""
        documents = []
        for url in urls:
            try:
                documents.append(self.load(url))
            except Exception as e:
                print(f"Failed to load {url}: {e}")
        return documents


class ImageLoader:
    """Load and extract text from images using OCR."""
    
    def __init__(self, language: str = "eng"):
        """
        Initialize image loader.
        
        Args:
            language: Tesseract language code
        """
        self.language = language
    
    def load(self, file_path: str) -> Document:
        """
        Load an image and extract text using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Document object with extracted content
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("Please install pytesseract and pillow")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        img = Image.open(file_path)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(img, lang=self.language)
        
        return Document(
            content=text.strip(),
            metadata={
                "size": img.size,
                "format": img.format,
                "file_name": os.path.basename(file_path)
            },
            source=file_path,
            doc_type="image"
        )
    
    def extract_with_vision_model(
        self,
        file_path: str,
        prompt: str = "Describe this image in detail."
    ) -> Document:
        """
        Extract information from image using vision-language model.
        
        Args:
            file_path: Path to the image file
            prompt: Prompt for the vision model
            
        Returns:
            Document object with model description
        """
        import base64
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        client = OpenAI()
        
        # Read and encode image
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine image type
        ext = os.path.splitext(file_path)[1].lower()
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
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        description = response.choices[0].message.content
        
        return Document(
            content=description,
            metadata={
                "file_name": os.path.basename(file_path),
                "prompt": prompt,
                "model": "gpt-4o"
            },
            source=file_path,
            doc_type="image"
        )


class AudioLoader:
    """Load and transcribe audio files."""
    
    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None
    ):
        """
        Initialize audio loader.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code (None for auto-detection)
        """
        self.model_size = model_size
        self.language = language
        self._model = None
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                import whisper
            except ImportError:
                raise ImportError("Please install openai-whisper: pip install openai-whisper")
            self._model = whisper.load_model(self.model_size)
        return self._model
    
    def load(self, file_path: str) -> Document:
        """
        Load and transcribe an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Document object with transcribed content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        model = self._load_model()
        
        # Transcribe
        result = model.transcribe(
            file_path,
            language=self.language
        )
        
        # Format transcript with timestamps
        segments = result.get("segments", [])
        formatted_transcript = []
        
        for segment in segments:
            start = self._format_timestamp(segment["start"])
            end = self._format_timestamp(segment["end"])
            text = segment["text"].strip()
            formatted_transcript.append(f"[{start} - {end}] {text}")
        
        return Document(
            content="\n".join(formatted_transcript),
            metadata={
                "language": result.get("language", "unknown"),
                "duration": segments[-1]["end"] if segments else 0,
                "file_name": os.path.basename(file_path)
            },
            source=file_path,
            doc_type="audio"
        )
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def transcribe_with_api(self, file_path: str) -> Document:
        """
        Transcribe audio using OpenAI Whisper API.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Document object with transcribed content
        """
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        client = OpenAI()
        
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
        
        return Document(
            content=transcript.text,
            metadata={
                "language": transcript.language,
                "duration": transcript.duration,
                "file_name": os.path.basename(file_path)
            },
            source=file_path,
            doc_type="audio"
        )


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    return text.strip()


if __name__ == "__main__":
    # Example usage
    print("Document Loaders initialized successfully!")
    print("Available loaders: PDFLoader, WebLoader, ImageLoader, AudioLoader")

