# Vision RAG Chat System

## 🎯 Overview

The `/app/chat` module provides an intelligent question-answering system that uses document page summaries to select relevant pages, then analyzes those specific images with vision AI to answer user questions.

**🆕 NEW: Defect Detection System** - Specialized service for high-quality defect detection in PDF documents using GPT-4/GPT-5 with optimized image processing. See [DEFECT_DETECTION_GUIDE.md](DEFECT_DETECTION_GUIDE.md) for details.

## 🏗️ Architecture

```
User Question
     ↓
PageSelector (uses summaries to select relevant pages)
     ↓
ChatAgent (analyzes selected page images with LLM)
     ↓
Answer + Source Pages + Cost
```

### Cost-Effective Approach

Instead of sending all pages to the vision model:
1. **First pass**: Uses text-based summaries (cheap) to identify relevant pages
2. **Second pass**: Only sends selected page images to vision model (expensive)

This can reduce costs by 70-90% compared to analyzing all pages!

## 📁 Module Structure

```
app/chat/
├── __init__.py           # Module exports
├── page_selector.py      # Selects relevant pages using summaries
├── chat_agent.py         # Orchestrates Q&A with vision analysis
└── chat_service.py       # High-level service interface
```

## 🚀 Running the Streamlit UI

### Start the app:

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit
streamlit run main.py
```

The UI will open in your browser at `http://localhost:8501`

## 💻 Using the UI

### 1. Upload a PDF
- Click "Browse files" in the sidebar
- Select a PDF file
- Click "🚀 Process Document"
- Wait for processing (converts pages to images and analyzes with AI)

### 2. Ask Questions
- Type your question in the text input
- Adjust "Max Pages" if needed (default: 5)
- Click "🔍 Ask Question"
- View the answer and see which pages were used

### 3. Features
- **Smart Page Selection**: Automatically selects relevant pages
- **Conversation History**: Keep track of all Q&A
- **Cost Tracking**: Monitor API usage
- **Multiple Documents**: Switch between uploaded documents
- **Document Details**: View page summaries and metadata

## 🔧 Programmatic Usage

### Basic Usage

```python
import asyncio
from app.chat.chat_service import ChatService

async def main():
    # Initialize service
    service = ChatService()
    
    # Ask a question
    result = await service.ask(
        document_id="doc_abc123",
        question="What is the candidate's education background?"
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Pages used: {result['page_numbers']}")
    print(f"Cost: ${result['total_cost']:.4f}")

asyncio.run(main())
```

### Advanced: Multi-turn Conversation

```python
import asyncio
from app.chat.chat_service import ChatService

async def main():
    service = ChatService()
    
    questions = [
        "What is this document about?",
        "Who is the author?",
        "What are the main findings?",
        "What methodology was used?"
    ]
    
    results = await service.conversation(
        document_id="doc_abc123",
        questions=questions,
        max_pages_per_question=3
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nQ{i}: {questions[i-1]}")
        print(f"A{i}: {result['answer']}")
        print(f"Pages: {result['page_numbers']}")

asyncio.run(main())
```

### List Available Documents

```python
from app.chat.chat_service import ChatService

service = ChatService()
documents = service.list_documents()

for doc in documents:
    print(f"- {doc['name']} ({doc['page_count']} pages)")
    print(f"  ID: {doc['id']}")
    print(f"  Has summaries: {doc['has_summaries']}")
```

## 🎛️ Configuration Options

### Page Selection

```python
result = await service.ask(
    document_id="doc_abc123",
    question="What is this about?",
    max_pages=5,           # Maximum pages to select
    use_all_pages=False    # Set True to skip selection
)
```

### Custom Provider

```python
from app.providers.factory import create_provider
from app.chat.chat_service import ChatService

# Use custom provider
provider = create_provider(
    api_key="your-key",
    base_url="https://custom-endpoint.com",
    model="gpt-4-vision"
)

service = ChatService(provider=provider)
```

## 💰 Cost Management

### Track Costs

```python
service = ChatService()

# Ask multiple questions
await service.ask(doc_id, "Question 1")
await service.ask(doc_id, "Question 2")

# Check total cost
total = service.get_total_cost()
print(f"Session cost: ${total:.4f}")

# Reset counter
service.reset_cost()
```

### Cost Breakdown

Each query returns:
- `selection_cost`: Cost of selecting pages (text-only)
- `analysis_cost`: Cost of vision analysis (images)
- `total_cost`: Combined cost

## 🔍 How It Works

### 1. Page Selection (`page_selector.py`)

```python
# Builds context from page summaries
pages_context = """
Document: CV.pdf
Total pages: 3

Page 1: Personal information and contact details
Page 2: Work experience at tech companies
Page 3: Education and certifications
"""

# Asks LLM which pages are relevant
# Returns: [2, 3] for question about work experience
```

### 2. Vision Analysis (`chat_agent.py`)

```python
# Sends selected page images + question to vision model
messages = [
    {"role": "system", "content": "You are a document analyst..."},
    {"role": "user", "content": [
        {"type": "text", "text": "Question: ..."},
        {"type": "image", "image_path": "page_2.jpg"},
        {"type": "image", "image_path": "page_3.jpg"}
    ]}
]
```

## 📊 Example Results

```python
{
    "answer": "The candidate has 5 years of experience as a Software Engineer at Google (2019-2024), where they led the development of cloud infrastructure tools.",
    "selected_pages": [<Page 2>, <Page 3>],
    "page_numbers": [2, 3],
    "total_cost": 0.0234,
    "selection_cost": 0.0004,  # Cheap text analysis
    "analysis_cost": 0.0230     # Vision analysis of 2 pages
}
```

## 🛠️ Troubleshooting

### "Document not found"
- Ensure the document has been processed and analyzed
- Check document ID with `service.list_documents()`

### "No page summaries"
- The document needs to be analyzed first
- Process the document with `VisionAnalysisService`

### High costs
- Reduce `max_pages` parameter
- Use more specific questions
- Enable page selection (don't use `use_all_pages=True`)

## 📝 Best Practices

1. **Specific Questions**: More specific questions → better page selection → lower costs
2. **Reasonable Max Pages**: Start with 3-5 pages, increase if needed
3. **Pre-analyze Documents**: Always run vision analysis before asking questions
4. **Monitor Costs**: Check costs regularly with `get_total_cost()`

## 🎓 Example Use Cases

### CV/Resume Analysis
```python
questions = [
    "What is the candidate's total years of experience?",
    "What programming languages do they know?",
    "What is their education background?",
    "Have they worked at any FAANG companies?"
]
```

### Research Paper Analysis
```python
questions = [
    "What is the main research question?",
    "What methodology was used?",
    "What are the key findings?",
    "What are the limitations mentioned?"
]
```

### Invoice/Form Processing
```python
questions = [
    "What is the total amount?",
    "What is the invoice date?",
    "Who is the vendor?",
    "What items are included?"
]
```

## 🚀 Next Steps

- Try the Streamlit UI: `streamlit run main.py`
- Explore the API in Python scripts
- Adjust `max_pages` to balance cost vs accuracy
- Build your own custom workflows

