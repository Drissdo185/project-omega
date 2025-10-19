```text
# Directory structure
ai_pdf_qa/
│
├── app/
│   ├── ingest/
│   │   ├── pdf_reader.py          # Read and extract text from PDF
│   │   ├── chunker.py             # Chunk PDF text into structured JSON
│   │
│   ├── classifier/
│   │   ├── labeler.py             # AI classifier (HR / IT / Other)
│   │
│   ├── storage/
│   │   ├── json_store.py          # Manage JSON files and global index
│   │
│   ├── search/
│   │   ├── searcher.py            # Local fuzzy search for context retrieval
│   │
│   ├── chat/
│   │   ├── manager.py             # Chat manager + prompt logic
│   │
│   ├── api/
│   │   ├── functions.py           # Exposed callable API functions
│   │
│   ├── utils/
│   │   ├── azure_openai_client.py # Wrapper for Azure OpenAI client
│   │   ├── tokenizer.py           # Token counting & chunking helpers
│   │
│   ├── ui/
│   │   ├── streamlit_app.py       # Streamlit UI frontend
│   │
│   └── __init__.py
│
├── data/
│   ├── HR/                        # HR JSON documents
│   ├── IT/                        # IT JSON documents
│   └── Other/                     # Uncategorized documents
│
├── requirements.txt               # Dependencies
├── README.md                      # This file
```

### Environment Setup
1️⃣ Create and activate virtual environment
```text
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)
```


### Install dependencies
```text
pip install -r requirements.txt
```

### Set up environment variables

# Create .env file (copy from .env.example):

```text
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo
AZURE_OPENAI_API_VERSION=2024-10-01-preview
CHUNK_SIZE_TOKENS=800
CHUNK_OVERLAP_TOKENS=60
BATCH_SAVE_SIZE=20
```