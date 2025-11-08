# 📄 Vision-Based PDF AI Assistant

> An intelligent PDF analysis system powered by Vision AI that can understand, analyze, and answer questions about PDF documents.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?logo=streamlit)](https://streamlit.io/)

---

## 🌟 Features

### 📊 **Intelligent Document Analysis**
- **Vision-Based Processing**: Converts PDF pages to images for advanced AI analysis
- **Smart Table Detection**: Automatically identifies and extracts tables from documents
- **Chart Recognition**: Detects and analyzes charts, graphs, and visualizations
- **Automatic Partitioning**: Handles large documents (>20 pages) with intelligent page grouping

### 💬 **Interactive Q&A System**
- **Context-Aware Responses**: AI understands document context to provide accurate answers
- **Page Selection Intelligence**: Automatically identifies relevant pages for each question
- **Confidence Scoring**: Shows confidence level (high/medium/low) for each answer
- **Chat History**: Maintains conversation context for follow-up questions

### 🎯 **Smart Architecture**
- **Two-Tier Strategy**: 
  - Small documents (≤20 pages): Direct page analysis
  - Large documents (>20 pages): Partition-based approach for better performance
- **Token Optimization**: Dynamic model selection based on document complexity
- **Efficient Caching**: Processed documents stored for quick access

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI-compatible API key
- 4GB+ RAM recommended

### 1. Clone Repository
```bash
git clone https://github.com/Drissdo185/project-omega.git
cd project-omega
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your API key
# OPENAI_API_KEY=your-api-key-here
```

### 3. Launch Application
```bash
# Build and start with Docker
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### 4. Access Application
Open your browser and navigate to:
```
http://localhost:8080
```

---

## 📖 Usage Guide

### Step 1: Upload PDF
- Click "Choose a PDF file" button
- Select your PDF document
- Supported format: PDF only

### Step 2: Process Document
- Click "🚀 Process Document with AI"
- Wait for the analysis to complete
- View document statistics (pages, tables, charts)

### Step 3: Ask Questions
- Type your question in the chat input
- AI will select relevant pages automatically
- Get detailed answers with source page references

### Example Questions:
```
❓ What is the main topic of this document?
❓ Can you summarize the key findings in the tables?
❓ What trends are shown in the charts?
❓ What are the main conclusions?
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                       │
│            (User Interface Layer)                   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────┐
│              Main Application                       │
│         (Orchestration & Flow Control)              │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────┴──────┐    ┌─────────┴──────────┐
│  PDF → Image │    │   AI Vision        │
│  Processor   │───▶│   Analyzer         │
│              │    │                    │
│ - PyMuPDF    │    │ - Table Detection  │
│ - Pillow     │    │ - Chart Detection  │
│ - Auto       │    │ - Summarization    │
│   Partition  │    │ - Metadata Extract │
└──────────────┘    └─────────┬──────────┘
                              │
                    ┌─────────┴──────────┐
                    │  Page Selection    │
                    │      Agent         │
                    │                    │
                    │ - Smart Retrieval  │
                    │ - Context Building │
                    │ - Q&A Generation   │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴──────────┐
                    │   OpenAI Client    │
                    │                    │
                    │ - Vision API       │
                    │ - Chat API         │
                    │ - Custom Endpoint  │
                    └────────────────────┘
```

### Data Flow

1. **PDF Upload** → User uploads PDF through Streamlit interface
2. **Image Conversion** → PDF pages converted to high-quality JPEG images
3. **Vision Analysis** → AI analyzes each page for content, tables, and charts
4. **Metadata Storage** → Results stored in JSON format with partitions
5. **Question Input** → User asks questions about the document
6. **Smart Retrieval** → System selects most relevant pages
7. **Answer Generation** → AI generates detailed answers with confidence scores

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | API key for AI services | - | ✅ Yes |
| `FLEX_RAG_DATA_LOCATION` | Data storage path | `/app/app/flex_rag_data_location` | No |
| `STREAMLIT_SERVER_PORT` | Application port | `8501` | No |
| `STREAMLIT_SERVER_ADDRESS` | Bind address | `0.0.0.0` | No |

### API Configuration

**Custom API Endpoint:**
```python
base_url = "https://aiportalapi.stu-platform.live/use"
```

**Models Used:**
- `Gemini-2.5-Flash` - Small documents & Q&A
- `Gemini-2.5-Flash` - Large documents

---

## 📦 Docker Commands

### Basic Operations
```bash
# Start application
docker-compose up -d

# Stop application
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build
```

### Maintenance
```bash
# Check container status
docker-compose ps

# View resource usage
docker stats vision-gpt5-pdf-assistant

# Access container shell
docker exec -it vision-gpt5-pdf-assistant bash

# Clean up everything
docker-compose down -v
docker system prune -a
```

---

## 📁 Project Structure

```
.
├── app/
│   ├── ai/
│   │   ├── openai.py              # OpenAI client wrapper
│   │   ├── vision_analyzer.py     # Vision-based page analysis
│   │   └── page_selection_agent.py # Smart page selection & Q&A
│   ├── processors/
│   │   ├── pdf_to_image.py        # PDF to image conversion
│   │   └── document.py            # Document data models
│   └── flex_rag_data_location/    # Processed documents storage
│       ├── documents/
│       │   └── {doc_id}/
│       │       ├── metadata.json
│       │       ├── partition_summary.json
│       │       └── pages/
│       └── cache/
├── main.py                        # Streamlit application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Docker Compose configuration
├── .env.example                   # Environment template
└── README.md                      # This file
```

---

## 🎨 Screenshots

### Main Interface
![Main Interface](https://via.placeholder.com/800x400/4A90E2/FFFFFF?text=PDF+Upload+Interface)

### Document Analysis
![Analysis](https://via.placeholder.com/800x400/7CB342/FFFFFF?text=Document+Analysis+View)

### Q&A Interaction
![Q&A](https://via.placeholder.com/800x400/FFA726/FFFFFF?text=Interactive+Q%26A+System)

---

## 🔒 Security Best Practices

- ✅ API keys stored in `.env` (never committed to git)
- ✅ `.dockerignore` prevents sensitive files in images
- ✅ Volume mounts for data persistence
- ✅ Health checks for container monitoring
- ✅ Resource limits in production mode

---

## 🐛 Troubleshooting

### Common Issues

**Problem:** Container unhealthy
```bash
# Check logs
docker-compose logs -f

# Verify API key
docker exec vision-gpt5-pdf-assistant printenv OPENAI_API_KEY
```

**Problem:** Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8080:8501"  # Use different port
```

**Problem:** Out of memory
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory
```

**Problem:** Build fails
```bash
# Clean Docker cache
docker system prune -a -f

# Rebuild from scratch
docker-compose build --no-cache
```

---

## 📚 API Documentation

### OpenAI Client

```python
from app.ai.openai import OpenAIClient

# Initialize client
client = OpenAIClient(api_key="your-key")

# Vision completion
response = await client.vision_completion(
    text_prompt="Analyze this image",
    images=[base64_image],
    model="Gemini-2.5-Flash"
)

# Chat completion
response = await client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="Gemini-2.5-Flash"
)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Clone repository
git clone https://github.com/Drissdo185/project-omega.git

# Install dependencies
pip install -r requirements.txt

# Run locally (without Docker)
streamlit run main.py
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Drissdo185**
- GitHub: [@Drissdo185](https://github.com/Drissdo185)
- Repository: [project-omega](https://github.com/Drissdo185/project-omega)

---

## 🙏 Acknowledgments

- **Streamlit** - Beautiful web framework for ML/AI apps
- **OpenAI** - Powerful AI models and APIs
- **PyMuPDF** - Excellent PDF processing library
- **Docker** - Containerization platform

---

## 📊 Performance

- **Processing Speed**: ~2-3 seconds per page
- **Memory Usage**: ~2-4GB for typical documents
- **Supported Size**: Up to 100+ pages per document
- **Concurrent Users**: Depends on server resources

---

## 🔮 Roadmap

- [ ] Multi-language support
- [ ] Batch document processing
- [ ] Export answers to PDF/DOCX
- [ ] Advanced chart analysis
- [ ] Document comparison feature
- [ ] API endpoints for programmatic access

---

<div align="center">

**Made with ❤️ using Streamlit, Docker, and AI**

[⬆ Back to Top](#-vision-based-pdf-ai-assistant)

</div>
