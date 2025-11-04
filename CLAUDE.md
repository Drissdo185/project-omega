# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Vision-based PDF RAG (Retrieval-Augmented Generation) system** built with Streamlit. It converts PDFs to images and uses OpenAI's vision models to analyze document content, extract tables/charts, and answer questions.

### Core Architecture

The system follows a **three-phase processing pipeline**:

1. **PDF → Images**: Convert PDF pages to JPEG images using PyMuPDF (main.py:42-98, app/processors/pdf_to_image.py)
2. **Vision Analysis**: AI analyzes each page to extract summaries, tables, and charts (app/ai/vision_analyzer.py)
3. **Question Answering**: Agent selects relevant pages and generates answers (app/ai/page_selection_agent.py)

### Document Size Strategy

The system handles documents differently based on size:

**Small Documents (≤20 pages)**:
- Direct page-level analysis
- All pages stored in `metadata.json`
- Q&A selects from all pages directly

**Large Documents (>20 pages)**:
- **Auto-partitioning**: Documents divided into partitions of 20 pages each (app/processors/pdf_to_image.py:62-97)
- **Two-tier metadata**:
  - `metadata.json` - Full page-level details
  - `partition_summary.json` - Aggregated partition summaries with tables/charts
- **Two-step Q&A**: First selects top 2 relevant partitions, then selects specific pages within them (app/ai/page_selection_agent.py:401-448)

This partitioning strategy is critical for handling large documents efficiently and staying within token limits.

### Data Model Hierarchy

```
Document
├── pages: List[Page]
│   ├── page_number, image_path, summary
│   ├── partition_id (for large docs)
│   ├── tables: List[TableInfo]
│   └── charts: List[ChartInfo]
└── partitions: List[Partition] (>20 pages only)
    ├── partition_id
    ├── page_range: (start, end)
    └── summary
```

See `app/processors/document.py` for complete data model definitions.

### Directory Structure

```
flex_rag_data_location/documents/{doc_id}/
├── pages/
│   ├── page_1.jpg
│   ├── page_2.jpg
│   └── ...
├── metadata.json          # Full document metadata with all pages
└── partition_summary.json # Large docs only: partition-level aggregation
```

## Development Commands

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run Streamlit app
streamlit run main.py
```

### Key Environment Variables

- `OPENAI_API_KEY` - Required for AI vision analysis and Q&A
- `FLEX_RAG_DATA_LOCATION` - Data storage location (defaults to `/flex_rag_data_location`)

## Important Implementation Details

### Vision Analyzer (app/ai/vision_analyzer.py)

The `VisionAnalyzer.analyze_document()` method orchestrates the entire analysis:

1. **Step 1**: Analyzes each page individually (`_analyze_single_page`)
   - Extracts page summary
   - Detects and summarizes tables
   - Detects and summarizes charts/graphs
   - Uses `model_small` (gpt-4o-mini)

2. **Step 2** (large docs only): Analyzes each partition (`_analyze_partition_batch`)
   - Samples up to 10 pages per partition
   - Generates partition-level summary
   - Uses `model_large` (gpt-5-mini) with low detail

3. **Step 3**: Saves `metadata.json`

4. **Step 4** (large docs only): Generates `partition_summary.json`

### Page Selection Agent (app/ai/page_selection_agent.py)

**Small Document Strategy** (`_select_pages_small_doc`):
- Reviews all page summaries with metadata
- Selects top 5 most relevant pages
- Uses `model_small`

**Large Document Strategy** (`_select_pages_large_doc`):
- **Phase 1**: Select top 2 partitions from `partition_summary.json` (`_select_partitions_from_summary`)
  - Uses partition summaries + aggregated table/chart metadata
  - Uses `model_large`
- **Phase 2**: Select top 5 pages within chosen partitions (`_select_pages_within_partitions`)
  - Uses page summaries + metadata
  - Uses `model_large`

This two-phase approach is essential for scaling to large documents.

### Model Selection (app/ai/openai.py)

```python
model_small = "gpt-4o-mini-2024-07-18"  # ≤20 pages
model_large = "gpt-5-mini-2025-08-07"   # >20 pages
```

The system uses different models based on document complexity.

## Code Organization

- **main.py** - Streamlit UI and orchestration
- **app/processors/** - PDF processing and document data models
  - `pdf_to_image.py` - PDF → image conversion with auto-partitioning
  - `document.py` - All data classes (Document, Page, Partition, TableInfo, ChartInfo, etc.)
- **app/ai/** - AI vision analysis and Q&A
  - `openai.py` - OpenAI client wrapper
  - `vision_analyzer.py` - Vision-based page/partition analysis
  - `page_selection_agent.py` - Intelligent page selection and Q&A

## Common Workflows

### Adding a New Feature to Analysis

1. Update data model in `app/processors/document.py` if needed
2. Modify vision prompt in `app/ai/vision_analyzer.py:_analyze_single_page`
3. Update JSON parsing to handle new fields
4. Ensure `to_dict()`/`from_dict()` methods support new fields

### Changing Partition Size

Modify `partition_size` parameter in `VisionPDFProcessor.__init__()` (currently 20 pages). This affects:
- Partition creation logic
- `partition_summary.json` structure
- Page selection agent behavior

### Modifying Q&A Strategy

Edit `PageSelectionAgent` methods:
- Adjust `max_pages_to_analyze` (currently 5)
- Adjust `max_partitions` (currently 2)
- Modify selection prompts for different ranking criteria

## Current Git Status

Branch: `feature/summary_agent`

Modified files:
- `app/ai/vision_analyzer.py` - Vision analysis implementation
- `app/processors/pdf_to_image.py` - PDF processor with partitioning
- `main.py` - Streamlit UI updates
- `app/__init__.py` - New file

Recent focus: Building partition-level summarization for large documents.
