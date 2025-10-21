# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Omega** is an AI-powered PDF Question Answering system with **dual-mode architecture**:

### Vision-Based RAG (New - Recommended)
**Location**: [flex-rag/](flex-rag/) directory

- Converts PDF pages to images using PyMuPDF
- Vision models directly analyze page images (no embeddings/vector DB needed)
- Adaptive query processing with task planning and page selection
- Stores: Images + Metadata in `flex_rag_data/`
- **Status**: ✅ **Fully implemented and ready to use**

See [flex-rag/README.md](flex-rag/README.md) for complete documentation.

### Text-Based RAG (Legacy)
**Location**: [app/](app/) directory

- Extracts text from PDFs using pdfplumber
- Classifies documents into HR/IT/Other categories
- Chunks text into ~800 token sections with fuzzy search (rapidfuzz)
- Provides strict numbered citation answers using Azure OpenAI
- **Status**: Maintained for backward compatibility

**Note**: Both systems can run simultaneously (dual-mode).

## Current Architecture (Text-Based Implementation)

### Core Pipeline Flow

1. **PDF Ingestion** → Extract text from PDFs using pdfplumber
2. **Classification** → Classify documents as HR, IT, or Other using Azure OpenAI or keyword fallback
3. **Chunking** → Split documents into ~800 token chunks with 60 token overlap
4. **Storage** → Save as JSON files in category folders (data/HR/, data/IT/, data/Other/)
5. **Query Processing** → Classify user query → Search relevant folder → Generate strict answer with numbered citations

### Directory Structure

```
app/
├── api/           # Function registry for external integrations
├── chat/          # Core message processing and answer generation
├── classifier/    # Document/query classification (HR/IT/Other)
├── ingest/        # PDF reading and text chunking
├── search/        # Fuzzy search over stored JSON sections
├── storage/       # JSON persistence and index management
├── utils/         # Azure OpenAI client, tokenizer
└── ui/            # Streamlit web interface (main.py entry point)

data/              # Document storage organized by category
├── HR/            # HR-classified documents as JSON
├── IT/            # IT-classified documents as JSON
├── Other/         # Other documents as JSON
└── index.json     # Master index of all documents

logs/
└── call_log.json  # Function call timeline for debugging
```

### Key Components

**AzureOpenAIClient** ([app/utils/azure_openai_client.py](app/utils/azure_openai_client.py))
- Wrapper for Azure OpenAI chat completions
- Supports function calling mode
- Gracefully degrades if env vars not set

**Document Classifier** ([app/classifier/labeler.py](app/classifier/labeler.py))
- Uses Azure OpenAI with zero-temperature for deterministic classification
- Falls back to keyword matching if API unavailable
- Returns label (HR/IT/Other) and confidence score

**Search Engine** ([app/search/searcher.py](app/search/searcher.py))
- Uses rapidfuzz for fuzzy matching between query and section content/titles
- Searches only within the classified folder
- Returns top-k sections with scores

**Chat Manager** ([app/chat/manager.py](app/chat/manager.py))
- End-to-end pipeline: classify → search → answer
- **Strict answer format**: Returns numbered citations `[1] excerpt` or exactly "I don't know"
- Logs every step to logs/call_log.json for debugging
- Validates LLM output to ensure numbered format compliance

**JSON Storage** ([app/storage/json_store.py](app/storage/json_store.py))
- Documents stored as JSON: `{title, sections: [{id, title, content, tokens}], metadata}`
- Central index.json tracks all documents with paths, labels, section counts

## Vision-Based Architecture (Implemented)

The vision-based RAG system is fully implemented in the `flex-rag/` directory.

### Directory Structure
```
flex-rag/
├── core/
│   └── config.py          # Configuration management
├── models/
│   ├── document.py        # Document, Page data models
│   └── query.py           # QueryResult model
├── storage/
│   ├── base.py            # BaseStorage interface
│   ├── local.py           # LocalStorage (filesystem)
│   └── memory.py          # MemoryStorage (testing)
├── processors/
│   ├── base.py            # BaseProcessor interface
│   ├── pdf.py             # PDFProcessor using PyMuPDF
│   └── factory.py         # ProcessorFactory pattern
├── providers/
│   ├── base.py            # BaseProvider interface
│   ├── openai.py          # OpenAI vision integration
│   ├── anthropic.py       # Anthropic Claude vision
│   └── factory.py         # ProviderFactory pattern
└── ai/
    ├── summarizer.py      # Document summarization
    ├── agent.py           # Adaptive RAG agent
    ├── page_selector.py   # Vision-based page selection
    ├── synthesizer.py     # Response synthesis
    └── prompts.py         # Centralized prompt management
```

### Vision-Based Processing Flow

**Add Document:**
```
PDF file
  → PDFProcessor (PyMuPDF)
  → Convert pages to JPEG images (90% quality)
  → Vision model summarizes ALL pages
  → Save images + metadata.json to storage
```

**Query Document:**
```
User question
  → Task Planning (break down complex queries)
  → Page Selection (vision model picks relevant pages from images)
  → Analysis (vision model reads selected page images)
  → Re-evaluation (adaptive loop if more context needed)
  → Synthesis (combine findings into answer)
```

### Vision-Based Data Storage

Documents stored as:
```
flex_rag_data/
└── documents/
    └── doc_abc123/
        ├── metadata.json      # Document info + summary
        └── pages/
            ├── page_1.jpg     # Page images (JPEG 90%)
            ├── page_2.jpg
            └── ...
```

**metadata.json structure:**
```json
{
  "id": "doc_abc123",
  "name": "research_paper.pdf",
  "page_count": 10,
  "status": "ready",
  "summary": "This paper discusses...",
  "pages": [
    {
      "page_number": 1,
      "image_path": "/path/to/page_1.jpg",
      "width": 1200,
      "height": 1600
    }
  ]
}
```

### Key Differences from Current Implementation

| Aspect | Current (Text-Based) | Target (Vision-Based) |
|--------|---------------------|----------------------|
| **Input** | Text extraction (pdfplumber) | Page images (PyMuPDF) |
| **Search** | Fuzzy text matching (rapidfuzz) | Vision model page selection |
| **Storage** | JSON with text chunks | Images + metadata |
| **Context** | Text snippets | Full page images |
| **Model** | Text-only LLM | Vision-capable LLM |
| **Embeddings** | None (fuzzy search) | None (direct vision analysis) |

### Migration Considerations

When implementing the vision-based architecture:

1. **No embedding/vector DB required** - Vision models directly analyze images, eliminating need for Pinecone or other vector stores
2. **Storage overhead** - Images require more disk space than text
3. **Latency** - Vision API calls may be slower than text-only
4. **Cost** - Vision model tokens are typically more expensive
5. **Accuracy on tables/charts** - Vision models excel at visual content that text extraction misses
6. **Adaptive querying** - New agent-based approach with re-evaluation loops for complex queries

## Development Commands

### Running the Application

```bash
# Start Streamlit UI (main entry point)
python main.py

# Alternative: direct streamlit command
streamlit run app/ui/streamlit_app.py
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Required Azure OpenAI credentials in `.env`:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_DEPLOYMENT_NAME=your-deployment
AZURE_VERSION=2024-02-15-preview
```

Optional tuning parameters:
```
CHUNK_SIZE_TOKENS=800          # Default chunk size
CHUNK_OVERLAP_TOKENS=60        # Token overlap between chunks
BATCH_SAVE_SIZE=20             # Batch size for saving
```

### Testing Document Ingestion

To ingest a new PDF programmatically:

```python
from app.ingest.pdf_reader import extract_full_text
from app.ingest.chunker import chunk_text_and_save

text = extract_full_text("path/to/document.pdf")
out_path, num_sections, label = chunk_text_and_save(
    document_title="My Document",
    text=text,
    source_filename="document.pdf"
)
print(f"Saved to {out_path} with {num_sections} sections, classified as {label}")
```

## Key Design Decisions

### Strict Answer Format

The system enforces a strict citation format to prevent hallucination:
- Valid responses are ONLY numbered citations: `[1] excerpt from context 1\n[2] excerpt from context 2`
- If no relevant context exists, return exactly `I don't know`
- The LLM is explicitly instructed NOT to paraphrase or add information
- Output validation rejects any response that doesn't follow this format

See `ANSWER_PROMPT_TEMPLATE` in [app/chat/manager.py:14](app/chat/manager.py#L14) for the exact prompt.

### Category-Based Search

- Documents are classified once during ingestion
- User queries are classified at query time
- Search is restricted to the matching category folder
- This prevents mixing HR/IT contexts and improves precision

### Fuzzy Matching (Current) vs Vision-Based Search (Target)

**Current approach**: Uses rapidfuzz (edit distance) for text matching
- Simple, fast, no API costs
- Limited semantic understanding
- Misses visual content (tables, charts, diagrams)

**Target approach**: Vision models directly analyze page images
- No embeddings or vector DB needed (see Target Architecture section above)
- Understands visual content natively
- Adaptive page selection with re-evaluation
- Higher cost and latency, but better accuracy on complex documents

### Logging Architecture

Every function call in the chat pipeline is logged to `logs/call_log.json` with:
- Timestamp
- Function name
- Arguments
- Result summary

This enables debugging the full conversation flow in the Streamlit sidebar "Conversation Timeline".

## Common Tasks

### Adding a New Document Category

1. Update classification prompt in [app/classifier/labeler.py:7](app/classifier/labeler.py#L7)
2. Update keyword fallback in [app/classifier/labeler.py:31](app/classifier/labeler.py#L31)
3. Update folder enum in [app/api/functions.py:28](app/api/functions.py#L28) if using API
4. Create new data subfolder: `mkdir data/NewCategory`

### Adjusting Chunk Size

Set environment variables before running:
```bash
export CHUNK_SIZE_TOKENS=1000
export CHUNK_OVERLAP_TOKENS=100
python main.py
```

### Debugging Failed Queries

Check `logs/call_log.json` for the complete pipeline trace:
1. `classify_text_short` - was the query classified correctly?
2. `find_relevant_sections` - did search return relevant contexts?
3. `AzureOpenAIClient.call_completion` - did the LLM receive the right prompt?
4. `validate_output` - did the LLM response pass format validation?

### Changing the UI

The Streamlit interface is in [app/ui/streamlit_app.py](app/ui/streamlit_app.py). Key sections:
- Lines 104-184: Chat message rendering with colored citations
- Lines 204-233: Input handling and chat clearing
- Lines 236-249: Indexed files viewer

## Using Vision-Based RAG

### Quick Start

**1. Ingest a PDF Document:**
```bash
python3 -m flex_rag.ingest_document path/to/document.pdf --label HR
```

**2. Query a Document:**
```python
from flex_rag.providers.factory import ProviderFactory
from flex_rag.storage.local import LocalStorage
from flex_rag.ai.agent import AdaptiveRAGAgent

provider = ProviderFactory.create_provider('azure')
storage = LocalStorage()
agent = AdaptiveRAGAgent(provider, storage)

result = agent.query(
    query="What is the leave policy?",
    document_id="doc_123"
)

print(result.answer)
print(f"Pages: {result.get_unique_page_numbers()}")
print(f"Cost: ${result.total_cost:.4f}")
```

**3. Migrate Existing Documents:**
```bash
# Migrate all legacy text-based documents
python3 scripts/migrate_to_vision.py
```

### Key Files to Know

**Vision RAG:**
- [flex-rag/README.md](flex-rag/README.md) - Complete vision RAG documentation
- [flex-rag/ai/agent.py](flex-rag/ai/agent.py) - Main adaptive RAG agent
- [flex-rag/ingest_document.py](flex-rag/ingest_document.py) - Document ingestion script
- [app/chat/vision_manager.py](app/chat/vision_manager.py) - Vision chat manager integration
- [app/api/vision_api.py](app/api/vision_api.py) - FastAPI endpoints for vision RAG

**Configuration:**
- [flex-rag/core/config.py](flex-rag/core/config.py) - All vision RAG settings
- `.env` - Azure OpenAI credentials (required)

## Important Constraints

### Legacy Text-Based System
- **No hallucination**: The system will return "I don't know" rather than generate unsupported answers
- **Category isolation**: Searches only within classified folders (no cross-category blending)
- **Azure dependency**: Requires Azure OpenAI credentials; falls back to keyword classification only
- **JSON storage**: Not designed for high-scale production (consider migration to vector DB)

### Vision-Based System
- **Higher costs**: Vision API calls are 3-5x more expensive than text-only
- **Longer latency**: 10-30 seconds per query (vs 2-5 seconds for text-based)
- **Azure OpenAI required**: No fallback mode without vision API credentials
- **Storage overhead**: Images require 10-20x more disk space than text chunks
- **Better accuracy**: Handles tables, charts, diagrams that text extraction misses
