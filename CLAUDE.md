# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered PDF Q&A system that uses vision-based RAG (Retrieval Augmented Generation) 

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables in .env
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_DEPLOYMENT_NAME=gpt-35-turbo
AZURE_VERSION=2024-10-01-preview
CHUNK_SIZE_TOKENS=800
CHUNK_OVERLAP_TOKENS=60
BATCH_SAVE_SIZE=20
```

## Common Commands

```bash
# Run the Streamlit UI
streamlit run main.py

# Process a PDF with the vision processor
python examples/simple_usage.py path/to/document.pdf HR

# Run detailed example showing all features
python examples/process_pdf_example.py

# Run tests (when implemented)
pytest tests/

# Check for test files
ls -la tests/ingest
```

## Architecture

### Core Processing Pipeline

1. **Document Ingestion** (`app/ingest/`)
   - Factory pattern (`factory.py`) creates appropriate processor based on file type
   - Vision-based PDF processing (`pdf_vision.py`) converts pages to high-quality JPEG images using PyMuPDF
   - No text extraction - pure vision approach for better accuracy with complex layouts
   - Base processor interface (`base.py`) defines contract for all processors

2. **Document Classification** (`app/classifier/`)
   - `labeler.py` uses Azure OpenAI to classify documents as HR, IT, or Other
   - Falls back to keyword-based heuristics if API unavailable
   - Temperature=0 for deterministic results

3. **Storage Layer** (`app/storage/`)
   - **NEW**: `document_store.py` - manages vision-based documents in `flex_rag_data_location/` structure
     - Methods: `load_document()`, `load_index()`, `list_documents_by_folder()`, `delete_document()`
     - Handles document metadata, page images, and cached summaries
     - Global index at `flex_rag_data_location/documents/index.json`
   - **Legacy**: `json_store.py` - manages text-based JSON documents in `data/` directory
     - Still used by legacy search/chat components
     - Organized by category (data/HR, data/IT, data/Other)

4. **Search & Retrieval** (`app/search/`)
   - `searcher.py` performs fuzzy text matching using rapidfuzz library
   - Searches within classified folder for relevant sections
   - Ranks by fuzzy score, returns top_k results

5. **Chat Pipeline** (`app/chat/`)
   - `manager.py` orchestrates the complete Q&A workflow:
     1. Classify user query → determine folder (HR/IT/Other)
     2. Search for relevant contexts in that folder
     3. Generate strict numbered answer using Azure OpenAI
     4. Log all function calls to `logs/call_log.json`
   - Strict prompt template enforces [1][2][3] citation format
   - Returns "I don't know" if no relevant context found

6. **API Functions** (`app/api/`)
   - `functions.py` exposes three main callable functions:
     - `classify_document(text)` - classify text as HR/IT/Other
     - `search_folder(query, folder)` - search specific folder
     - `answer_from_context(session_state, query, top_k)` - end-to-end Q&A

7. **Data Models** (`app/models/`)
   - `document.py` defines core domain models:
     - `Page`: represents single page as image with metadata
     - `Document`: container with pages, status, folder classification
     - `DocumentStatus`: enum (PROCESSING/READY/ERROR)
   - All models support `to_dict()` and `from_dict()` serialization

8. **Utilities** (`app/utils/`)
   - `azure_openai_client.py` - wrapper for Azure OpenAI client with graceful degradation
   - `tokenizer.py` - token counting and chunking helpers

### UI Layer

- `main.py` - Streamlit application providing chat interface
- Session state tracks conversation history, contexts, sources, and function call logs
- Sidebar displays conversation timeline from `logs/call_log.json`
- Modern chat bubble UI with numbered source citations
- Auto-scrolling and proper input clearing

### Data Organization

The system now uses a structured directory format for document storage:

```
flex_rag_data_location/                    # Root storage directory
├── documents/
│   ├── doc_abc123/                 # Document 1
│   │   ├── metadata.json          # Document metadata and page info
│   │   └── pages/
│   │       ├── page_1.jpg
│   │       ├── page_2.jpg
│   │       └── ...
│   │
│   ├── doc_xyz789/                 # Document 2
│   │   ├── metadata.json
│   │   └── pages/
│   │       ├── page_1.jpg
│   │       └── ...
│   │
│   └── index.json                  # Global document index
│
└── cache/                          # Optional cache
    └── summaries/
        └── doc_abc123_summary.txt

# Legacy structure (still supported by some components):
data/
├── HR/           # HR-classified JSON documents
├── IT/           # IT-classified JSON documents
├── Other/        # Uncategorized documents
└── index.json    # Global document index
```

## Key Implementation Details

### Vision-Based Processing
- PDFs converted to images at 2x scale for quality (configurable via `render_scale` parameter)
- Images resized to max 1400x1400 pixels to balance quality vs. token cost (configurable via `max_image_size`)
- JPEG quality 90% with optimization (configurable via `jpeg_quality`)
- **NEW**: Images saved to structured directory: `flex_rag_data_location/documents/{doc_id}/pages/`
- Each document gets unique ID generated from file path + timestamp MD5 hash
- Metadata stored in `metadata.json` alongside pages
- Global index updated automatically at `flex_rag_data_location/documents/index.json`
- Storage root is configurable (default: `./flex_rag_data_location`)

### Strict Answer Format
- System enforces numbered citation format: `[1] exact excerpt from context`
- Validation checks for proper `[n]` format in responses
- Falls back to "I don't know" for invalid outputs or empty contexts
- Temperature=0 for deterministic answers

### Function Call Logging
- All pipeline steps logged to both session state and `logs/call_log.json`
- Includes timestamps, function names, arguments, and result summaries
- UI displays last 50 entries in sidebar timeline
- Persisted to disk for debugging and audit trail

### Fuzzy Search Strategy
- Uses `rapidfuzz.fuzz.partial_ratio` for matching
- Searches both section content (first 1000 chars) and title
- Minimum score threshold of 10 (very permissive for recall)
- Falls back to filesystem scan if index missing entries

### Azure OpenAI Integration
- Client gracefully handles missing credentials (logs warning, returns empty responses)
- Supports function calling for agent-style workflows
- System messages separated from user prompts
- Error handling with loguru logging

## Development Notes

- All paths use `os.path.join()` for cross-platform compatibility
- Loguru used for structured logging throughout
- Pydantic models in `app/models/` for data validation
- Factory pattern allows easy extension to new document types (Word, Excel, etc.)
- Session state pattern (Streamlit) maintains conversation context without database

## Testing

Test fixtures should be placed in `tests/fixtures/`. Currently test infrastructure is minimal - tests should be added for:
- PDF vision processing (image quality, page count)
- Classification accuracy (HR vs IT vs Other)
- Search relevance and ranking
- Strict answer format validation
