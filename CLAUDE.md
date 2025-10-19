# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Omega** is a Vision-Based RAG (Retrieval-Augmented Generation) system for querying document content using visual representations of pages.

The system is designed to:
- Process documents as collections of page images
- Generate document summaries
- Answer queries by selecting relevant pages and providing answers with confidence scores

## Architecture

### Core Data Models (`models/`)

The system uses dataclass-based models with dictionary serialization:

- **Document** (`models/document.py`): Represents a multi-page document with lifecycle management
  - Status flow: `"processing"` → `"ready"` | `"failed"`
  - Contains Page objects, metadata, and document summary
  - Document IDs: generated as `"doc_{12-char-hex}"`

- **Page** (`models/page.py`): Represents a single document page
  - Stores page number, image path, and dimensions (width/height)

- **QueryResult** (`models/query.py`): Represents query answers
  - Contains answer text, selected page numbers, document ID
  - Optional confidence score and reasoning
  - Includes timestamp and extensible metadata

### Design Patterns

- All models implement `to_dict()` / `from_dict()` for JSON serialization
- Timestamps use UTC (`datetime.now(datetime.timezone.utc)`)
- Documents automatically update `updated_at` on mutations
- Document readiness check: `is_ready()` requires status `"ready"` and non-null summary

## Development

### Python Environment

Dependencies are listed in `requirement.txt` (currently empty - to be populated).

### Project Structure

```
models/          # Data models and domain objects
  document.py    # Document and Page models
  query.py       # QueryResult model
```

## Key Implementation Notes

- Document status must be managed through `update_status()` to ensure `updated_at` is maintained
- Pages are 1-indexed via `page_number` field
- Document summaries are required before querying (enforced by `is_ready()`)
- All models support metadata dictionaries for extensibility
