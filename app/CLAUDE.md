# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project Omega is a **vision-based document RAG (Retrieval-Augmented Generation) system** that processes PDF documents as high-quality images and uses multimodal AI models to analyze and answer questions about them. Unlike traditional text-extraction RAG systems, this approach preserves visual layout, formatting, and graphical elements.

## Development Setup

**Prerequisites:**
- Python 3.12+
- Virtual environment located at `/home/dtdat/Desktop/Project/project-omega/venv/`

**Environment Variables (in `/home/dtdat/Desktop/Project/project-omega/.env`):**
```bash
FLEX_RAG_DATA_LOCATION=./flex_rag_data_location  # Document storage location
OPENAI_BASE_URL=<custom-endpoint>                 # OpenAI-compatible API endpoint
OPENAI_MODEL=GPT-5                                # Model identifier
OPENAI_API_KEY=<your-api-key>                     # API authentication
```

**Installation:**
```bash
# Activate virtual environment
source ../venv/bin/activate

# Install dependencies
pip install -r ../requirements.txt
```

**Running Modules:**
```bash
# Process a PDF document
python -m app.processor.pdf_vision

# Test vision analysis
python -m app.ai.vision_analysis

# Run providers
python -m app.providers.openai
```

## Architecture

### Core Design Principles

1. **Vision-First Approach**: PDFs are converted to JPEG images (2x render scale, 90% quality) instead of extracting text. This preserves formatting, tables, charts, and visual context.

2. **Factory Pattern**: `ProcessorFactory` and `create_provider()` enable pluggable implementations for different file types and LLM providers.

3. **Async-First**: All I/O operations use `async`/`await` for concurrent processing.

4. **File-Based Storage**: Documents stored in structured directories (`documents/{doc_id}/pages/page_*.jpg`) with JSON metadata. No database required.

### Module Structure

```
app/
├── models/              # Core data models (Document, Page, QueryResult, AgentTask)
├── processor/           # Document processing pipeline (PDF → JPEG conversion)
├── providers/           # LLM provider abstraction (OpenAI-compatible)
├── storage/             # Document persistence (file-based with JSON index)
└── ai/                  # Vision-based analysis service
```

### Data Flow

```
PDF File → ProcessorFactory.create_processor()
         → VisionPDFProcessor.process()
         → Document (with Page[] images)
         → DocumentStore.add_document()
         → VisionAnalysisService.analyze_document()
         → DocumentAnalysis (JSON summary)
```

### Key Components

**1. Document Processing (`app/processor/`)**
- `ProcessorFactory`: Creates appropriate processor based on file type
- `VisionPDFProcessor`: Converts PDF to JPEG images using PyMuPDF (fitz)
- Configuration: `render_scale=2.0`, `jpeg_quality=90`, `max_image_size=(1400, 1400)`
- Output: `Document` object with `Page[]` containing image paths

**2. LLM Providers (`app/providers/`)**
- `BaseProvider`: Abstract interface for LLM communication
- `OpenAIProvider`: OpenAI-compatible API client supporting text and vision
- Factory function: `create_provider(provider_type="openai")`
- Methods: `process_text_messages()`, `process_multimodal_messages()`
- Tracks API costs via `get_last_cost()`

**3. Document Storage (`app/storage/`)**
- `DocumentStore`: Manages document CRUD operations
- Storage structure:
  ```
  {storage_root}/
  ├── index.json              # Global document index
  └── documents/
      └── {doc_id}/
          ├── metadata.json   # Document metadata
          └── pages/
              ├── page_0001.jpg
              ├── page_0002.jpg
              └── ...
  ```
- Methods: `add_document()`, `get_document()`, `get_all_documents()`, `delete_document()`

**4. Vision Analysis (`app/ai/`)**
- `VisionAnalysisService`: Analyzes document pages using vision models
- `PageAnalysis`: Per-page analysis with summary, topics, labels, confidence
- `DocumentAnalysis`: Synthesized analysis of entire document
- `PageLabel`: Categorization (content type, topics, categories)
- Analysis modes: `detailed=True` (2000 tokens) or quick (1000 tokens)
- Stores results in `{storage_root}/analysis/{doc_id}_analysis.json`

**5. Data Models (`app/models/`)**
- `Document`: Core document model with status tracking
  - Status: `PROCESSING`, `READY`, `ERROR`
  - Fields: id, name, page_count, folder (HR/IT/Other), pages, summary
- `Page`: Single page as JPEG with metadata (dimensions, path)
- `QueryResult`: Query response with page references
- `AgentTask`/`TaskPlan`: Agent orchestration models (experimental)
- `ConversationMessage`: Chat history tracking

### Document Categories

Built-in categorization system in `app/ai/analysis.py`:

**Categories:**
- `hr_policy` - HR policies, employee handbooks
- `it_manual` - IT documentation, technical manuals
- `financial` - Financial documents, reports
- `legal` - Legal documents, contracts
- `general` - Other documents

**Content Types:**
- `text_heavy` - Primarily textual content
- `tables_charts` - Data visualizations
- `mixed` - Combination of text and visuals
- `forms` - Structured forms
- `images` - Image-heavy documents

## Common Patterns

### Processing a New PDF
```python
from app.processor.factory import ProcessorFactory
from app.storage.document_store import DocumentStore

# Create processor
processor = ProcessorFactory.create_processor("document.pdf")

# Process document
document = await processor.process(
    file_path="document.pdf",
    doc_id="unique-doc-id",
    folder="HR"
)

# Store document
store = DocumentStore()
store.add_document(document)
```

### Analyzing a Document
```python
from app.ai.vision_analysis import VisionAnalysisService
from app.providers.factory import create_provider

# Initialize service
provider = create_provider("openai")
service = VisionAnalysisService(provider)

# Analyze document
analysis = await service.analyze_document(
    document=document,
    detailed=True
)

# Save analysis
service.save_analysis(document.id, analysis)
```

### Using LLM Provider
```python
from app.providers.factory import create_provider

provider = create_provider("openai")

# Text-only request
response = await provider.process_text_messages(
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1000,
    temperature=0.7
)

# Multimodal (vision) request
response = await provider.process_multimodal_messages(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_path", "image_path": "/path/to/image.jpg", "detail": "high"}
        ]
    }],
    max_tokens=2000
)

# Check cost
cost = provider.get_last_cost()
```

## Important Notes

### Known Issues

1. **Undefined Attributes in `vision_analysis.py`:**
   - Line 189: References `pa.labels.entities` but `PageLabel` has no `entities` field
   - Line 238: Uses `document_entities` parameter not defined in `DocumentAnalysis`
   - Line 301: References `has_sensitive_info` not in `PageLabel` schema
   - These may cause runtime errors during document analysis

2. **Missing Entry Points:**
   - No FastAPI application entry point (`main.py` or `app.py`)
   - No Streamlit frontend despite being in requirements
   - Modules must be run individually for testing

3. **Security Concern:**
   - `.env` file should be gitignored (it is) but may contain sensitive API keys
   - Never commit actual API keys to version control

### Migration from Text-Based to Vision-Based

This codebase recently migrated from text extraction to vision-based processing:
- **Removed:** Legacy modules (`/ingest/`, text-based search, old API, chat manager)
- **Added:** Vision analysis, provider abstraction, new models structure
- **Legacy dependencies still present:** pypdf, pdfplumber, tiktoken (can be removed)

### Testing

No test suite currently exists. When adding tests:
- Use `pytest` as the testing framework
- Mock the `BaseProvider` for analysis tests
- Use fixture PDFs for processor tests
- Test file I/O with temporary directories

### Code Style

- Use `@dataclass` for data models with `to_dict()`/`from_dict()` methods
- All I/O operations should be `async`
- Use `loguru` for logging (not `print()`)
- Type hints required for all function signatures
- Follow factory pattern for extensibility

### Storage Locations

- **Documents:** `{FLEX_RAG_DATA_LOCATION}/documents/{doc_id}/`
- **Analysis:** `{FLEX_RAG_DATA_LOCATION}/analysis/{doc_id}_analysis.json`
- **Index:** `{FLEX_RAG_DATA_LOCATION}/index.json`
- Default: `./flex_rag_data_location` (relative to project root)

## Project Context

**Current branch:** `feature-DatDT`
**Recent refactoring:** Complete rebuild of source code with focus on vision-based approach
**Status:** Active development, core modules functional but no frontend/API layer yet
