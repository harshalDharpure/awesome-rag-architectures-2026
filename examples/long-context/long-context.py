"""
Long-Context RAG Implementation

Leverages large context windows (100K+ tokens) for
full-document understanding without chunking.
"""

import os
import sys
from typing import Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv

from utils.loaders import PDFLoader, Document
from utils.llm import get_llm, LLMWrapper

load_dotenv()


@dataclass
class LongContextResult:
    """Result from long-context RAG."""
    question: str
    answer: str
    document_tokens: int
    total_tokens: int
    citations: list[str]


class LongContextRAG:
    """
    Long-Context RAG for full-document understanding.
    
    Uses models with large context windows to process
    entire documents without chunking-induced information loss.
    """
    
    # Token limits for different models
    MODEL_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-3.5-turbo": 16385,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "gemini-1.5-pro": 1000000,
    }
    
    def __init__(
        self,
        llm_model: str = "gpt-4o",
        max_context_tokens: Optional[int] = None,
        reserve_tokens: int = 4000,
        temperature: float = 0.3
    ):
        """
        Initialize Long-Context RAG.
        
        Args:
            llm_model: LLM model name
            max_context_tokens: Max tokens for document (uses model limit if None)
            reserve_tokens: Tokens reserved for prompt and response
            temperature: LLM temperature (lower for factual tasks)
        """
        self.llm_model = llm_model
        self.llm = get_llm(llm_model, temperature=temperature)
        
        # Set context limit
        model_limit = self.MODEL_LIMITS.get(llm_model, 128000)
        self.max_context_tokens = max_context_tokens or (model_limit - reserve_tokens)
        self.reserve_tokens = reserve_tokens
        
        print(f"Initialized with {llm_model} ({model_limit} token limit)")
        print(f"Max document tokens: {self.max_context_tokens}")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses rough approximation of 4 characters per token.
        For production, use tiktoken for accurate counts.
        """
        return len(text) // 4
    
    def _truncate_to_limit(
        self,
        document: str,
        max_tokens: int
    ) -> tuple[str, bool]:
        """
        Truncate document to fit within token limit.
        
        Args:
            document: Document text
            max_tokens: Maximum tokens
            
        Returns:
            Tuple of (truncated_text, was_truncated)
        """
        estimated_tokens = self._estimate_tokens(document)
        
        if estimated_tokens <= max_tokens:
            return document, False
        
        # Truncate to approximate character limit
        char_limit = max_tokens * 4
        truncated = document[:char_limit]
        
        # Try to end at a sentence boundary
        last_period = truncated.rfind('. ')
        if last_period > char_limit * 0.9:  # If we're at least 90% through
            truncated = truncated[:last_period + 1]
        
        truncated += "\n\n[Document truncated due to length...]"
        
        return truncated, True
    
    def _prioritize_sections(
        self,
        document: str,
        query: str,
        max_tokens: int
    ) -> str:
        """
        Prioritize document sections based on query relevance.
        
        For very long documents, this helps ensure relevant
        content appears early in the context.
        """
        # Split into sections (simple approach using double newlines)
        sections = document.split('\n\n')
        
        if len(sections) <= 3:
            return document
        
        # Score sections by query term overlap
        query_terms = set(query.lower().split())
        
        scored_sections = []
        for i, section in enumerate(sections):
            section_terms = set(section.lower().split())
            overlap = len(query_terms & section_terms)
            scored_sections.append((overlap, i, section))
        
        # Sort by relevance score (descending), then by original position
        scored_sections.sort(key=lambda x: (-x[0], x[1]))
        
        # Reassemble with high-relevance sections first
        result_parts = []
        current_tokens = 0
        
        for _, orig_idx, section in scored_sections:
            section_tokens = self._estimate_tokens(section)
            if current_tokens + section_tokens <= max_tokens:
                result_parts.append((orig_idx, section))
                current_tokens += section_tokens
        
        # Sort back to original order for coherence
        result_parts.sort(key=lambda x: x[0])
        
        return '\n\n'.join(section for _, section in result_parts)
    
    def query(
        self,
        question: str,
        document: str,
        system_prompt: Optional[str] = None,
        prioritize_sections: bool = True
    ) -> LongContextResult:
        """
        Answer a question using the full document context.
        
        Args:
            question: The question to answer
            document: The full document text
            system_prompt: Custom system prompt
            prioritize_sections: Whether to reorder sections by relevance
            
        Returns:
            LongContextResult with answer and metadata
        """
        # Estimate document tokens
        doc_tokens = self._estimate_tokens(document)
        
        # Prioritize sections if needed
        if prioritize_sections and doc_tokens > self.max_context_tokens:
            document = self._prioritize_sections(
                document, question, self.max_context_tokens
            )
        
        # Truncate if still too long
        document, was_truncated = self._truncate_to_limit(
            document, self.max_context_tokens
        )
        
        if was_truncated:
            print(f"⚠️ Document truncated from {doc_tokens} to ~{self.max_context_tokens} tokens")
        
        # Build prompts
        default_system = """You are an expert document analyst. Your task is to answer questions 
based on the provided document. Follow these guidelines:

1. Base your answer ONLY on information in the document
2. Cite specific sections or quotes when possible
3. If the document doesn't contain enough information, say so
4. Be comprehensive but concise
5. When citing, use [Section: X] or quote directly"""

        system = system_prompt or default_system
        
        user_prompt = f"""Document:
================================================================================
{document}
================================================================================

Question: {question}

Please provide a thorough answer based on the document above. Include specific citations where relevant."""

        # Generate response
        response = self.llm.generate(user_prompt, system_prompt=system)
        
        # Extract citations (simple extraction)
        citations = []
        import re
        citation_patterns = [
            r'\[Section: ([^\]]+)\]',
            r'"([^"]+)"',
            r"'([^']+)'"
        ]
        for pattern in citation_patterns:
            matches = re.findall(pattern, response.content)
            citations.extend(matches[:5])  # Limit to 5 citations
        
        return LongContextResult(
            question=question,
            answer=response.content,
            document_tokens=self._estimate_tokens(document),
            total_tokens=response.usage.get("total_tokens", 0),
            citations=citations[:5]
        )
    
    def query_multiple_documents(
        self,
        question: str,
        documents: list[dict],
        max_docs: int = 5
    ) -> LongContextResult:
        """
        Query across multiple documents.
        
        Args:
            question: The question to answer
            documents: List of dicts with 'content' and 'title'
            max_docs: Maximum documents to include
            
        Returns:
            LongContextResult
        """
        # Combine documents with headers
        combined = []
        total_tokens = 0
        
        for i, doc in enumerate(documents[:max_docs]):
            title = doc.get("title", f"Document {i+1}")
            content = doc["content"]
            
            doc_tokens = self._estimate_tokens(content)
            if total_tokens + doc_tokens > self.max_context_tokens:
                # Truncate this document
                remaining = self.max_context_tokens - total_tokens
                content, _ = self._truncate_to_limit(content, remaining)
            
            combined.append(f"=== {title} ===\n\n{content}")
            total_tokens += self._estimate_tokens(content)
            
            if total_tokens >= self.max_context_tokens:
                break
        
        full_document = "\n\n" + "="*80 + "\n\n".join(combined)
        
        return self.query(question, full_document, prioritize_sections=False)


class StructuredDocumentRAG(LongContextRAG):
    """
    Enhanced long-context RAG with document structure awareness.
    """
    
    def parse_document_structure(self, document: str) -> dict:
        """
        Parse document structure (headers, sections, etc.).
        
        Args:
            document: Document text
            
        Returns:
            Structured document representation
        """
        import re
        
        structure = {
            "title": "",
            "sections": [],
            "metadata": {}
        }
        
        lines = document.split('\n')
        current_section = {"header": "Introduction", "content": [], "level": 0}
        
        for line in lines:
            # Detect headers (markdown style)
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save current section
                if current_section["content"]:
                    structure["sections"].append({
                        "header": current_section["header"],
                        "content": '\n'.join(current_section["content"]),
                        "level": current_section["level"]
                    })
                
                # Start new section
                level = len(header_match.group(1))
                header = header_match.group(2)
                current_section = {"header": header, "content": [], "level": level}
                
                if level == 1 and not structure["title"]:
                    structure["title"] = header
            else:
                current_section["content"].append(line)
        
        # Don't forget last section
        if current_section["content"]:
            structure["sections"].append({
                "header": current_section["header"],
                "content": '\n'.join(current_section["content"]),
                "level": current_section["level"]
            })
        
        return structure
    
    def query_with_structure(
        self,
        question: str,
        document: str
    ) -> LongContextResult:
        """
        Query with structure-aware prompting.
        """
        structure = self.parse_document_structure(document)
        
        # Build table of contents
        toc = "Document Structure:\n"
        for i, section in enumerate(structure["sections"]):
            indent = "  " * section["level"]
            toc += f"{indent}- {section['header']}\n"
        
        # Enhanced prompt with structure
        system_prompt = f"""You are analyzing a structured document.

{toc}

Use this structure to navigate and cite specific sections in your answer.
When citing, reference section headers like [Section: Header Name]."""
        
        return self.query(question, document, system_prompt=system_prompt)


def main():
    """Example usage of Long-Context RAG."""
    
    # Sample long document (legal contract style)
    document = """
# SERVICE AGREEMENT

## 1. Parties

This Service Agreement ("Agreement") is entered into as of January 1, 2025, 
by and between TechCorp Inc., a Delaware corporation ("Provider"), and 
ClientCo LLC, a California limited liability company ("Client").

## 2. Services

### 2.1 Scope of Services

Provider agrees to provide Client with the following services:

a) Cloud Infrastructure Management
   - 24/7 monitoring and maintenance
   - Automatic scaling based on demand
   - Security patching and updates
   - Backup and disaster recovery

b) Technical Support
   - Priority email support (4-hour response)
   - Phone support during business hours
   - On-site support as needed (additional fees apply)

c) Consulting Services
   - Architecture review and recommendations
   - Performance optimization
   - Security assessments

### 2.2 Service Level Agreement

Provider commits to the following service levels:

- Uptime: 99.9% monthly availability
- Response Time: < 200ms for API calls (95th percentile)
- Support Response: < 4 hours for critical issues

Failure to meet these SLAs will result in service credits as described in Section 5.

## 3. Term and Termination

### 3.1 Initial Term

This Agreement shall commence on the Effective Date and continue for a period 
of twelve (12) months ("Initial Term").

### 3.2 Renewal

After the Initial Term, this Agreement will automatically renew for successive 
twelve (12) month periods unless either party provides written notice of 
non-renewal at least sixty (60) days prior to the end of the then-current term.

### 3.3 Termination for Cause

Either party may terminate this Agreement upon thirty (30) days written notice 
if the other party materially breaches any provision of this Agreement and fails 
to cure such breach within the notice period.

## 4. Fees and Payment

### 4.1 Service Fees

Client agrees to pay the following fees:

| Service Tier | Monthly Fee | Included Hours |
|--------------|-------------|----------------|
| Basic        | $5,000      | 40 hours       |
| Professional | $12,000     | 100 hours      |
| Enterprise   | $25,000     | Unlimited      |

### 4.2 Payment Terms

- Invoices issued monthly in advance
- Payment due within thirty (30) days of invoice date
- Late payments subject to 1.5% monthly interest
- Client responsible for all applicable taxes

### 4.3 Price Adjustments

Provider may increase fees upon sixty (60) days written notice, but such 
increases shall not exceed 5% annually.

## 5. Service Credits

### 5.1 Uptime Credits

If monthly uptime falls below the guaranteed 99.9%, Client is entitled to:

| Uptime | Credit |
|--------|--------|
| 99.0% - 99.9% | 10% of monthly fee |
| 95.0% - 99.0% | 25% of monthly fee |
| < 95.0% | 50% of monthly fee |

### 5.2 Credit Limitations

- Maximum credits per month: 50% of monthly fee
- Credits must be claimed within 30 days
- Credits applied to future invoices only

## 6. Confidentiality

### 6.1 Definition

"Confidential Information" means any non-public information disclosed by 
either party, including technical data, trade secrets, business plans, 
customer information, and financial data.

### 6.2 Obligations

Each party agrees to:
- Maintain confidentiality using reasonable care
- Not disclose to third parties without consent
- Use only for purposes of this Agreement
- Return or destroy upon termination

### 6.3 Exclusions

Confidential Information does not include information that:
- Is publicly available through no fault of receiving party
- Was known prior to disclosure
- Is independently developed
- Is required to be disclosed by law

## 7. Limitation of Liability

### 7.1 Cap on Liability

IN NO EVENT SHALL EITHER PARTY'S TOTAL LIABILITY EXCEED THE FEES PAID BY 
CLIENT IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.

### 7.2 Exclusion of Damages

NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, 
CONSEQUENTIAL, OR PUNITIVE DAMAGES, REGARDLESS OF THE CAUSE OF ACTION.

## 8. Governing Law

This Agreement shall be governed by the laws of the State of Delaware, 
without regard to conflicts of law principles.

## 9. Signatures

IN WITNESS WHEREOF, the parties have executed this Agreement as of the 
Effective Date.

TECHCORP INC.
By: ________________________
Name: Jane Smith
Title: CEO

CLIENTCO LLC
By: ________________________
Name: John Doe
Title: Managing Partner
"""

    # Initialize Long-Context RAG
    print("🚀 Initializing Long-Context RAG...")
    rag = LongContextRAG(
        llm_model="gpt-4o",
        temperature=0.3
    )
    
    # Test queries
    queries = [
        "What are the monthly fees for each service tier?",
        "What happens if the uptime falls below 99%?",
        "How can this agreement be terminated?",
        "What are the confidentiality obligations?"
    ]
    
    for question in queries:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print("=" * 60)
        
        result = rag.query(question, document)
        
        print(f"\n📊 Document tokens: ~{result.document_tokens}")
        print(f"📊 Total tokens used: {result.total_tokens}")
        
        if result.citations:
            print(f"\n📍 Citations found: {len(result.citations)}")
            for cite in result.citations[:3]:
                print(f"   - {cite[:50]}...")
        
        print("\n💡 Answer:")
        print("-" * 40)
        print(result.answer)


if __name__ == "__main__":
    main()

