# 🚀 Quick Start Guide

## One-Minute Setup

### 1. Prerequisites
```bash
✓ Docker installed
✓ 4GB+ RAM
✓ OpenAI-compatible API key
```

### 2. Setup & Run
```bash
# Clone & navigate
git clone https://github.com/Drissdo185/project-omega.git
cd project-omega

# Configure API key
cp .env.example .env
# Edit .env: Set OPENAI_API_KEY=your-key-here

# Launch
docker-compose up -d --build

# Access
# Open: http://localhost:8080
```

### 3. Usage
1. **Upload PDF** → Click "Choose a PDF file"
2. **Process** → Click "🚀 Process Document with AI"
3. **Ask Questions** → Type in chat: "What is this document about?"

---

## Quick Commands

```bash
# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Stop
docker-compose down

# Clean rebuild
docker-compose down -v && docker-compose up -d --build
```

---

## Troubleshooting

### Container unhealthy?
```bash
docker-compose logs -f
docker exec vision-gpt5-pdf-assistant printenv OPENAI_API_KEY
```

### Port 8080 in use?
Edit `docker-compose.yml`:
```yaml
ports:
  - "8888:8501"  # Use port 8888 instead
```

### Build fails?
```bash
docker system prune -a -f
docker-compose build --no-cache
```

---

## More Info
- Full Documentation: [README.md](README.md)
- Docker Guide: [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
