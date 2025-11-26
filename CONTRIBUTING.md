# Contributing to Awesome RAG Architectures 2025

First off, thank you for considering contributing to this project! 🎉

This document provides guidelines and steps for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers warmly
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

## 🤝 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Error messages** and stack traces
- **Code snippets** if applicable

### 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a **clear and descriptive title**
- Provide a **detailed description** of the proposed feature
- Explain **why this would be useful**
- Include **code examples** if applicable

### 🔧 Code Contributions

#### What to Contribute

- **New RAG architectures** (e.g., Self-RAG, CRAG, Speculative RAG)
- **Performance optimizations**
- **Bug fixes**
- **Documentation improvements**
- **Benchmark additions**
- **New utility functions**

#### Architecture Contributions

When adding a new RAG architecture:

1. Create a new folder in `/examples/`
2. Include:
   - `README.md` with theory and usage
   - `.py` implementation file
   - `.ipynb` notebook with examples
3. Add corresponding diagram in `/diagrams/`
4. Update main `README.md` with the new architecture

## 🛠️ Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment tool (venv, conda)

### Setup Steps

```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/awesome-rag-architectures-2025.git
cd awesome-rag-architectures-2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black isort flake8 pytest mypy

# Create .env file
cp .env.example .env
# Add your API keys to .env
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=utils --cov=examples

# Run specific test file
pytest tests/test_chunking.py
```

## 📐 Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Use Black for formatting
black .

# Use isort for import sorting
isort .

# Check with flake8
flake8 --max-line-length=100
```

#### Code Standards

```python
# Good: Descriptive variable names
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Bad: Cryptic names
em = ST("all-MiniLM-L6-v2")

# Good: Type hints
def retrieve_documents(query: str, top_k: int = 5) -> list[Document]:
    ...

# Good: Docstrings
def chunk_text(text: str, chunk_size: int = 512) -> list[str]:
    """
    Split text into chunks of specified size.
    
    Args:
        text: The input text to chunk.
        chunk_size: Maximum characters per chunk.
        
    Returns:
        List of text chunks.
    """
    ...
```

### Documentation Style

- Use Markdown for all documentation
- Include code examples
- Add Mermaid diagrams for architecture explanations
- Keep language clear and beginner-friendly

### Commit Messages

Follow conventional commits:

```
feat: add self-rag implementation
fix: resolve chunking edge case with empty documents
docs: update multi-hop README with performance notes
refactor: simplify embedding selection logic
test: add unit tests for hybrid search
```

## 📤 Pull Request Process

### Before Submitting

1. **Update documentation** if you changed functionality
2. **Add tests** for new features
3. **Run linting** and fix issues
4. **Test your changes** thoroughly
5. **Update CHANGELOG** if applicable

### PR Template

When creating a PR, include:

```markdown
## Description
[Describe your changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed my code
- [ ] Added necessary documentation
- [ ] Added tests (if applicable)
- [ ] All tests pass
- [ ] Updated README if needed
```

### Review Process

1. Submit PR to `main` branch
2. Maintainers will review within 48-72 hours
3. Address any feedback
4. Once approved, your PR will be merged

## 🏷️ Issue Labels

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `architecture` | New RAG architecture |
| `benchmark` | Benchmark-related |

## 🎖️ Recognition

Contributors will be:
- Listed in the README credits section
- Mentioned in release notes
- Given contributor badge on their profile

## 📞 Getting Help

- Open an issue for questions
- Join our Discord community (coming soon)
- Tag maintainers for urgent issues

---

Thank you for contributing! 🙏

Every contribution, no matter how small, makes this project better for everyone.

