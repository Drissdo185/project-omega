# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Omega** is a vision-based RAG (Retrieval-Augmented Generation) system for PDF document analysis. Unlike traditional text-based RAG systems, it converts PDF pages to images and uses vision models (GPT-4o, GPT-5) to analyze content, detect tables/charts, and answer questions. This approach preserves visual layout and handles complex documents better than text extraction.

## Core Architecture

### Multi-Stage Query Flow

The system uses **conditional 2-stage or 3-stage flows** based on document size:

**2-Stage Flow (≤20 pages):**
1. Page Selection: Analyze page summaries to select relevant pages (uses `gpt-4o-mini`)
2. Vision Analysis: Analyze selected page images to answer question (uses `gpt-4o-mini`)

**3-Stage Hierarchical Flow (>20 pages):**
1. Partition Selection: Select 1-2 relevant partitions from 5 total partitions (uses `gpt-5`)
2. Page Selection: Select 2-5 pages from chosen partitions only (uses `gpt-5`)
3. Vision Analysis: Analyze selected page images to answer question (uses `gpt-5`)

This is implemented in `app/chat/chat_agent.py:45-141`.

### Document Processing Pipeline

1. **PDF to Images** (`app/processor/pdf_vision.py`):
   - Converts PDF pages to JPEG images using PyMuPDF (fitz)
   - Stores in `{storage_root}/documents/{doc_id}/pages/`
   - Default render scale: 1.5x, max size: 1024x1024px, quality: 75%

2. **Vision Analysis** (`app/ai/vision_analysis.py`):
   - Analyzes each page image with vision model to extract:
     - Page summary (concise description)
     - Tables: `{table_id, title, summary}`
     - Charts: `{chart_id, title, chart_type, summary}`
   - Processes pages concurrently (10 pages at a time by default)
   - For large documents (>20 pages): creates 5 partitions with summaries

3. **Metadata Storage** (`app/storage/document_store.py`):
   - Saves complete analysis to `{doc_id}/metadata.json`
   - Maintains global index at `documents/index.json`

### Data Models (`app/models/`)

Key models that drive the system:

- **Document**: Root entity with pages, partitions, status, summary
  - `is_large_document()`: Returns true if >20 pages (triggers 3-stage flow)
  - `has_partitions()`: Checks if partition summaries exist
- **Page**: Image path, dimensions, summary, tables[], charts[]
- **Partition**: For large docs, groups ~(page_count/5) pages each
- **TableInfo/ChartInfo**: Detected visual elements with summaries

### Provider System (`app/providers/`)

Abstraction layer for LLM providers:

- **BaseProvider**: Abstract interface defining `process_text_messages()` and `process_multimodal_messages()`
- **OpenAIProvider**: Implementation using OpenAI Responses API
  - Supports two model configurations via env vars:
    - `OPENAI_MODEL_2STAGE` (default: `gpt-4o-mini`)
    - `OPENAI_MODEL_3STAGE` (default: `gpt-5-mini-2025-08-07`)
  - Encodes images as base64 for API calls
  - Methods: `get_model_2stage()`, `get_model_3stage()`, `get_last_cost()`

### Selection Components (`app/chat/`)

- **PageSelector** (`page_selector.py`): Uses page summaries to select relevant pages
  - Sends page summaries (text-only) to LLM to decide relevance
  - Can filter by partitions for 3-stage flow
  - Returns JSON array of page numbers
- **PartitionSelector** (`partition_selector.py`): For large docs, selects 1-2 partitions
  - Uses partition summaries (text-only) to narrow search space
  - Reduces pages to analyze from hundreds to ~40-80
- **ChatAgent** (`chat_agent.py`): Orchestrates the complete query flow
  - Method: `answer_question(document, question, max_pages, use_all_pages)`
  - Returns dict with: answer, selected_pages, costs breakdown, selected_partitions

## Environment Setup

### Required Environment Variables

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here

# Model Selection (optional, defaults shown)
OPENAI_MODEL_2STAGE=gpt-4o-mini           # For small docs ≤20 pages
OPENAI_MODEL_3STAGE=gpt-5-mini-2025-08-07 # For large docs >20 pages

# Storage Location (optional)
FLEX_RAG_DATA_LOCATION=./flex_rag_data_location
```

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Common Development Tasks

### Running the Application

No main entry point exists in the current codebase (main.py was deleted). The system is designed to be used as a library. Typical usage pattern:

```python
from app.processor.pdf_vision import VisionPDFProcessor
from app.ai.vision_analysis import VisionAnalysisService
from app.chat.chat_agent import ChatAgent
from app.providers.openai import OpenAIProvider

# 1. Process PDF to images
processor = VisionPDFProcessor()
document = await processor.process("path/to/file.pdf")

# 2. Analyze pages with vision model
provider = OpenAIProvider()
analyzer = VisionAnalysisService(provider)
document = await analyzer.analyze_document(document)

# 3. Query the document
agent = ChatAgent(provider)
result = await agent.answer_question(document, "What are the main findings?")
```

### Storage Structure

All data stored under `FLEX_RAG_DATA_LOCATION` (default: `./flex_rag_data_location/`):

```
flex_rag_data_location/
├── documents/
│   ├── index.json                    # Global document index
│   ├── doc_abc123/
│   │   ├── metadata.json             # Complete document metadata
│   │   └── pages/
│   │       ├── page_1.jpg
│   │       ├── page_2.jpg
│   │       └── ...
│   └── doc_xyz789/
│       └── ...
└── cache/
    └── summaries/
```

## Important Implementation Details

### Cost Tracking

Each component tracks API costs via `provider.get_last_cost()`:
- **partition_cost**: Cost of partition selection (3-stage only)
- **selection_cost**: Cost of page selection
- **analysis_cost**: Cost of final vision analysis
- **total_cost**: Sum of all stages

Cost tracking is critical for optimizing the multi-stage approach.

### Concurrency Control

`VisionAnalysisService.analyze_document()` processes pages in batches of 10 concurrently. Adjust `max_concurrent` parameter for different API rate limits or performance needs.

### Partition Creation

For documents >20 pages, partitions are created automatically during analysis:
- Fixed 5 partitions (`num_partitions=5` in `_create_partitions()`)
- Equal page distribution: partition size ≈ page_count/5
- Each partition gets a summary synthesized from its page summaries

### Image Processing Parameters

Default settings in `VisionPDFProcessor.__init__()`:
- `render_scale=1.5`: PDF rendering quality multiplier
- `jpeg_quality=75`: JPEG compression quality (0-100)
- `max_image_size=(1024, 1024)`: Maximum dimensions before resizing

Adjust these to balance quality vs. API token costs.

## Code Patterns

### Async/Await

All LLM calls and document processing use async patterns. Always use `await` when calling:
- `processor.process()`
- `analyzer.analyze_document()`, `analyzer.analyze_page()`
- `agent.answer_question()`
- `provider.process_text_messages()`, `provider.process_multimodal_messages()`

### Error Handling

The codebase uses try/except with logger from `loguru`. Failures typically:
- Log error with `logger.error()`
- Return sensible fallbacks (e.g., empty lists, first page)
- Don't crash the entire pipeline

### Model Selection

When calling provider methods, pass explicit `model` parameter to control which model is used:
```python
# Use 2-stage model (cheaper)
await provider.process_text_messages(messages, model=provider.get_model_2stage())

# Use 3-stage model (more capable)
await provider.process_text_messages(messages, model=provider.get_model_3stage())
```

The ChatAgent automatically selects the appropriate model based on document size.

## Key Files Reference

- `app/models/document.py:180-182` - `is_large_document()` threshold (20 pages)
- `app/chat/chat_agent.py:68-87` - Conditional 2-stage vs 3-stage routing
- `app/ai/vision_analysis.py:166-182` - Partition creation for large docs
- `app/chat/page_selector.py:26-104` - Page selection logic with partition filtering
- `app/chat/partition_selector.py:26-96` - Partition selection for large docs
- `app/processor/pdf_vision.py:52-139` - PDF to image conversion
- `app/providers/openai.py:46-86` - Text-only LLM calls
- `app/providers/openai.py:88-127` - Multimodal (text+image) LLM calls

## Git Workflow

Current branch: `feature/summary_agent`
Recent changes involve chat agents, page/partition selectors, and vision analysis updates.
