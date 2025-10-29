# Vision RAG System - Architecture

## System Overview

```mermaid
graph TB
    subgraph "User Interface"
        UI[Streamlit UI<br/>main.py]
    end

    subgraph "Chat Layer"
        CS[ChatService<br/>chat_service.py]
        CA[ChatAgent<br/>chat_agent.py]
        PS[PageSelector<br/>page_selector.py]
    end

    subgraph "Processing Layer"
        PDF[VisionPDFProcessor<br/>pdf_vision.py]
        VAS[VisionAnalysisService<br/>vision_analysis.py]
    end

    subgraph "Provider Layer"
        PF[Provider Factory<br/>factory.py]
        OAI[OpenAI Provider<br/>openai.py]
        BP[Base Provider<br/>base.py]
    end

    subgraph "Storage Layer"
        DS[DocumentStore<br/>document_store.py]
        FS[(File System<br/>metadata.json<br/>page images)]
    end

    subgraph "Data Models"
        DM[Document Model<br/>document.py]
        AM[Analysis Model<br/>analysis.py]
        QM[Query Model<br/>query.py]
    end

    subgraph "External Services"
        LLM[LLM API<br/>GPT-5/Vision Models]
    end

    UI -->|1. Upload PDF| PDF
    UI -->|2. Ask Question| CS
    
    PDF -->|Convert to Images| FS
    PDF -->|Create Document| DM
    
    VAS -->|Analyze Pages| OAI
    VAS -->|Save Metadata| DS
    
    CS -->|Load Document| DS
    CS -->|Delegate Query| CA
    
    CA -->|Select Pages| PS
    CA -->|Analyze Images| OAI
    
    PS -->|Filter by Summary| OAI
    
    DS -->|Read/Write| FS
    
    OAI -->|API Calls| LLM
    OAI -.->|Implements| BP
    
    PF -->|Create| OAI
    
    DM -->|Used by| PDF
    DM -->|Used by| VAS
    DM -->|Used by| CA
    
    style UI fill:#e1f5ff
    style CS fill:#fff4e1
    style CA fill:#fff4e1
    style PS fill:#fff4e1
    style PDF fill:#e8f5e9
    style VAS fill:#e8f5e9
    style DS fill:#f3e5f5
    style FS fill:#f3e5f5
    style LLM fill:#ffebee
```

## Data Flow - Document Processing

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Proc as PDF Processor
    participant VA as Vision Analysis
    participant Provider as OpenAI Provider
    participant LLM as GPT-5 API
    participant Store as Document Store
    participant FS as File System

    User->>UI: Upload PDF
    UI->>Proc: process(pdf_path)
    
    Proc->>Proc: Generate doc_id
    Proc->>FS: Create directories
    
    loop For each page
        Proc->>Proc: Render page to image
        Proc->>Proc: Optimize & resize
        Proc->>FS: Save page_N.jpg
    end
    
    Proc->>UI: Return Document object
    
    UI->>VA: analyze_document(doc)
    
    loop For each page
        VA->>Provider: process_multimodal_messages<br/>(image + prompt)
        Provider->>Provider: Encode image to base64
        Provider->>LLM: Vision API call
        LLM->>Provider: Page summary
        Provider->>VA: Summary + cost
        VA->>VA: Update page.summary
    end
    
    VA->>Store: save_document_metadata()
    Store->>FS: Write metadata.json
    Store->>FS: Update index.json
    
    VA->>UI: Document analyzed ✓
```

## Data Flow - Question Answering

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant CS as Chat Service
    participant CA as Chat Agent
    participant PS as Page Selector
    participant Provider as OpenAI Provider
    participant LLM as GPT-5 API
    participant Store as Document Store

    User->>UI: Ask question
    UI->>CS: ask(doc_id, question)
    CS->>Store: load_document(doc_id)
    Store->>CS: Document with summaries
    
    CS->>CA: answer_question(doc, question)
    
    rect rgb(255, 244, 225)
        Note over CA,LLM: Phase 1: Smart Page Selection
        CA->>PS: select_relevant_pages(doc, question)
        PS->>PS: Build context from summaries
        PS->>Provider: process_text_messages<br/>(summaries + question)
        Provider->>LLM: Text-only API call (cheap)
        LLM->>Provider: [1, 3, 5]
        Provider->>PS: Page numbers + cost
        PS->>CA: Selected pages
    end
    
    rect rgb(232, 245, 233)
        Note over CA,LLM: Phase 2: Vision Analysis
        CA->>CA: Build multimodal prompt
        CA->>Provider: process_multimodal_messages<br/>(images + question)
        Provider->>Provider: Encode images to base64
        Provider->>LLM: Vision API call (expensive)
        LLM->>Provider: Detailed answer
        Provider->>CA: Answer + cost
    end
    
    CA->>CS: Result with answer, pages, cost
    CS->>UI: Display result
    UI->>User: Show answer + metadata
```

## Component Architecture

```mermaid
graph LR
    subgraph "Frontend Layer"
        A[Streamlit UI]
    end
    
    subgraph "Service Layer"
        B[Chat Service]
    end
    
    subgraph "Agent Layer"
        C[Chat Agent]
        D[Page Selector]
    end
    
    subgraph "Processing Layer"
        E[PDF Processor]
        F[Vision Analysis]
    end
    
    subgraph "Provider Layer"
        G[Provider Factory]
        H[OpenAI Provider]
    end
    
    subgraph "Storage Layer"
        I[Document Store]
    end
    
    subgraph "Data Layer"
        J[(File System)]
    end
    
    A --> B
    B --> C
    B --> I
    C --> D
    C --> H
    D --> H
    A --> E
    E --> F
    F --> H
    F --> I
    G --> H
    I --> J
    H --> K[External LLM API]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#fff4e1
    style D fill:#fff4e1
    style E fill:#e8f5e9
    style F fill:#e8f5e9
    style G fill:#ffe0b2
    style H fill:#ffe0b2
    style I fill:#f3e5f5
    style J fill:#f3e5f5
    style K fill:#ffebee
```

## Storage Structure

```mermaid
graph TB
    ROOT[flex_rag_data_location/]
    
    ROOT --> DOCS[documents/]
    ROOT --> CACHE[cache/]
    
    DOCS --> INDEX[index.json<br/>Global document index]
    DOCS --> DOC1[doc_abc123/]
    DOCS --> DOC2[doc_def456/]
    
    DOC1 --> META1[metadata.json<br/>Document + Analysis]
    DOC1 --> PAGES1[pages/]
    
    PAGES1 --> P1[page_1.jpg]
    PAGES1 --> P2[page_2.jpg]
    PAGES1 --> P3[page_3.jpg]
    
    CACHE --> SUMM[summaries/]
    SUMM --> S1[doc_abc123_summary.txt]
    
    style ROOT fill:#e8f5e9
    style DOCS fill:#fff4e1
    style CACHE fill:#fff4e1
    style INDEX fill:#e1f5ff
    style META1 fill:#e1f5ff
    style PAGES1 fill:#f3e5f5
    style P1 fill:#ffebee
    style P2 fill:#ffebee
    style P3 fill:#ffebee
```

## Cost Optimization Strategy

```mermaid
graph TD
    Q[User Question]
    
    Q --> S1[Phase 1: Summary Analysis<br/>Text-only LLM call]
    S1 --> COST1[Cost: ~$0.0001 - $0.001]
    
    S1 --> FILTER[Filter Relevant Pages<br/>e.g., 3 out of 20 pages]
    
    FILTER --> S2[Phase 2: Vision Analysis<br/>Multi-image LLM call]
    S2 --> COST2[Cost: ~$0.01 - $0.05]
    
    COST1 --> TOTAL[Total Cost: $0.01 - $0.05]
    COST2 --> TOTAL
    
    ALT[Alternative: No Filtering]
    ALT --> S3[Vision Analysis of All Pages]
    S3 --> COST3[Cost: ~$0.20 - $0.50+]
    
    TOTAL -.->|Saves 80-90%| COST3
    
    style Q fill:#e1f5ff
    style S1 fill:#e8f5e9
    style S2 fill:#fff4e1
    style S3 fill:#ffebee
    style TOTAL fill:#c8e6c9
    style COST3 fill:#ffcdd2
```

## Agent Decision Flow

```mermaid
flowchart TD
    START[User Question Received]
    
    START --> ANALYZE[Analyze Question Complexity]
    
    ANALYZE --> SIMPLE{Simple Question?<br/>e.g., What is X?}
    ANALYZE --> COMPLEX{Complex Question?<br/>e.g., Compare X and Y}
    ANALYZE --> BROAD{Broad Question?<br/>e.g., Summarize}
    
    SIMPLE -->|1-2 pages| SELECT1[Select Specific Pages<br/>Target: 1-2 pages]
    COMPLEX -->|3-5 pages| SELECT2[Select Multiple Pages<br/>Target: 3-5 pages]
    BROAD -->|5+ pages| SELECT3[Select Comprehensive<br/>Target: 5+ pages]
    
    SELECT1 --> LLM1[LLM Text Analysis<br/>of Summaries]
    SELECT2 --> LLM1
    SELECT3 --> LLM1
    
    LLM1 --> PAGES[Selected Page Numbers]
    
    PAGES --> VISION[Vision Analysis<br/>of Selected Images]
    
    VISION --> ANSWER[Generate Answer]
    
    ANSWER --> METADATA[Return Answer + Metadata<br/>Pages used, Cost, etc.]
    
    style START fill:#e1f5ff
    style ANALYZE fill:#fff4e1
    style SIMPLE fill:#c8e6c9
    style COMPLEX fill:#fff9c4
    style BROAD fill:#ffccbc
    style LLM1 fill:#e1f5ff
    style VISION fill:#f3e5f5
    style ANSWER fill:#c8e6c9
```

## Key Design Patterns

### 1. **Factory Pattern**
- `ProcessorFactory` creates appropriate document processors
- `ProviderFactory` creates LLM provider instances

### 2. **Strategy Pattern**
- `BaseProvider` interface allows swapping LLM providers
- Different processing strategies for different file types

### 3. **Repository Pattern**
- `DocumentStore` abstracts storage operations
- Clean separation between business logic and persistence

### 4. **Two-Phase Optimization**
- Phase 1: Cheap text analysis for filtering
- Phase 2: Expensive vision analysis on selected pages
- Result: 80-90% cost reduction

### 5. **Async/Await**
- All I/O operations are async
- Better performance for API calls and file operations

## Technology Stack

```mermaid
mindmap
  root((Vision RAG))
    Frontend
      Streamlit
      Python asyncio
    Processing
      PyMuPDF fitz
      Pillow PIL
      JPEG optimization
    AI/ML
      OpenAI API
      GPT-5 Vision
      Token usage tracking
    Storage
      JSON metadata
      File system
      Document indexing
    Models
      Pydantic
      Dataclasses
      Enums
    Utils
      python-dotenv
      loguru
      pathlib
```

## Performance Characteristics

| Operation | Time | Cost | Notes |
|-----------|------|------|-------|
| PDF Processing | ~2-5s/page | $0 | Local operation |
| Page Analysis | ~3-8s/page | ~$0.01/page | Vision API call |
| Page Selection | ~1-2s | ~$0.001 | Text-only API call |
| Question Answer | ~5-15s | ~$0.02-0.05 | Depends on pages selected |

## Scalability Considerations

- **Horizontal**: Multiple documents processed independently
- **Caching**: Page summaries cached after first analysis
- **Cost Control**: Smart page selection reduces API costs
- **Storage**: Efficient JPEG compression for images
- **Async**: Non-blocking I/O for better throughput

