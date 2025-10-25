# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Vision-based RAG (Retrieval-Augmented Generation)** system that processes PDF documents by converting them to images and analyzing them using vision-enabled LLMs. Unlike traditional text-based RAG systems, this approach preserves visual layout, formatting, tables, charts, and other visual elements.

## Architecture

### Core Pipeline

1. **PDF Processing** (`app/processor/pdf_vision.py`)
   - Converts PDF pages to high-quality JPEG images using PyMuPDF (fitz)
   - Stores images in structured directory: `flex_rag_data_location/documents/{doc_id}/pages/`
   - Generates unique document IDs and maintains metadata

2. **Vision Analysis** (`app/ai/vision_analysis.py`)
   - Analyzes document images using vision-capable LLMs
   - Performs page-level analysis with content extraction, labeling, and topic identification
   - Aggregates page analyses into document-level summaries
   - Tracks API costs for each analysis

3. **LLM Providers** (`app/providers/`)
   - Abstract provider interface (`base.py`) supporting both text and multimodal messages
   - OpenAI-compatible provider implementation (`openai.py`)
   - Handles base64 image encoding for vision requests
   - Configurable via environment variables

4. **Storage** (`app/storage/document_store.py`)
   - Manages document metadata and analysis results
   - Maintains global index at `flex_rag_data_location/documents/index.json`
   - Stores document metadata and analysis in `{doc_id}/metadata.json`

### Data Models

- **Document** (`app/models/document.py`): Represents a processed PDF with pages and metadata
- **Page**: Individual page with image path and dimensions
- **DocumentAnalysis** (`app/ai/analysis.py`): Complete analysis with category, summary, and page analyses
- **PageAnalysis**: Per-page analysis with summary, labels, and extracted data
- **PageLabel**: Content type, topics, language, and confidence metadata

### Storage Structure

```
flex_rag_data_location/
├── documents/
│   ├── index.json                    # Global document index
│   ├── {doc_id}/
│   │   ├── metadata.json             # Document + analysis metadata (merged)
│   │   └── pages/
│   │       ├── page_1.jpg
│   │       ├── page_2.jpg
│   │       └── ...
└── cache/
    └── summaries/                    # Cached summaries
```

## Environment Setup

### Required Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://aiportalapi.stu-platform.live/use  # Custom endpoint
OPENAI_MODEL=gpt-5                                          # Default model
FLEX_RAG_DATA_LOCATION=./flex_rag_data_location           # Data storage root
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Common Commands

### Run the Main Pipeline

```bash
# Activate virtual environment first
source venv/bin/activate

# Run main.py to process a PDF
python main.py
```

Note: Edit `main.py` to change the PDF file path (line 14).

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add New Dependencies

```bash
pip install <package>
pip freeze > requirements.txt
```

## Key Design Decisions

### Vision-First Approach

- **No text extraction**: PDFs are converted directly to images without OCR or text extraction
- **Preserves visual context**: Tables, charts, formatting, and layout are maintained
- **Multimodal analysis**: LLMs analyze images directly using vision capabilities

### Metadata Storage

- Analysis results are merged into the document's `metadata.json` file (not separate files)
- This keeps document metadata and analysis together in a single source of truth
- The `VisionAnalysisService._save_analysis()` method merges analysis fields with existing metadata

### Provider Abstraction

- `BaseProvider` interface allows swapping LLM providers
- Currently implements OpenAI-compatible endpoints
- Supports custom base URLs for alternative API endpoints
- Handles both text-only and multimodal (vision) requests

### Cost Tracking

- Each API call's token usage is tracked
- Costs are calculated based on GPT-5 pricing (configurable in `openai.py:162-172`)
- Total analysis cost is stored in document metadata

## Important File Locations

- Entry point: `main.py`
- PDF processor: `app/processor/pdf_vision.py:89` (process method)
- Vision analysis: `app/ai/vision_analysis.py:121` (analyze_document method)
- Provider factory: `app/providers/factory.py:39` (create_provider_from_env)
- Document storage: `app/storage/document_store.py`
- Data models: `app/models/document.py` and `app/ai/analysis.py`

## Development Notes

### Processing Configuration

Default settings in `VisionPDFProcessor.__init__()`:
- `render_scale: 2.0` - High-quality rendering
- `jpeg_quality: 90` - Optimized compression
- `max_image_size: (1400, 1400)` - Max dimensions for LLM vision

### Analysis Prompts

Vision analysis prompts are built in:
- `VisionAnalysisService._build_detailed_analysis_prompt()` - Full analysis
- `VisionAnalysisService._build_quick_analysis_prompt()` - Quick summary

### Adding New Document Processors

1. Inherit from `BaseProcessor` (`app/processor/base.py`)
2. Implement `process(file_path)` and `supports(file_path)` methods
3. Register in `app/processor/factory.py` (if using factory pattern)
