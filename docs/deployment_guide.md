# Deployment Guide - Integrated Healthcare Platform

## Prerequisites

- Python 3.10+
- Docker 24.0+
- Docker Compose 2.20+
- Google Gemini API key
- 4GB+ RAM recommended

## Local Development

### 1. Clone and Setup
```bash
git clone <repo-url>
cd 09_integrated_healthcare_platform

# Using existing venv (preferred)
source ../venv/bin/activate  # Linux/Mac
# OR
..\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Or set environment variable:
```bash
export GEMINI_API_KEY=your_key_here  # Linux/Mac
set GEMINI_API_KEY=your_key_here    # Windows
```

### 3. Initialize Vector Database
```bash
# Build ChromaDB with medical knowledge
python scripts/build_vector_db.py

# Ingest medical documents
python scripts/ingest_documents.py
```

### 4. Run Application
```bash
# Streamlit (main interface)
streamlit run app.py --server.port 8501

# Or FastAPI (REST API)
uvicorn src.api.main:app --reload --port 8000
```

## Docker Deployment

### Build and Run
```bash
# Build image
docker build -t healthcare-platform:latest .

# Run container
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key healthcare-platform:latest
```

### Docker Compose (Multi-Service)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Production Deployment

### Option 1: Docker Swarm
```bash
docker stack deploy -c docker-compose.yml healthcare
```

### Option 2: Kubernetes
```bash
kubectl apply -f deployment/kubernetes/
```

### Option 3: Cloud Platform
- **AWS**: ECS with Fargate or EC2
- **Azure**: Container Instances or AKS
- **GCP**: Cloud Run or GKE

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| GEMINI_API_KEY | Yes | - | Google Gemini API key |
| CHROMA_DB_PATH | No | ./data/chroma_db | ChromaDB persistent path |
| AUDIT_DB_PATH | No | ./data/audit.db | SQLite audit log path |
| LOG_LEVEL | No | INFO | Logging level |
| STREAMLIT_PORT | No | 8501 | Streamlit server port |
| MAX_TOKENS | No | 2048 | LLM max output tokens |

## Monitoring

Access monitoring dashboards:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Backup & Recovery

```bash
# Backup databases
cp -r data/chroma_db backups/chroma_db_$(date +%Y%m%d)
cp data/audit.db backups/audit_$(date +%Y%m%d).db

# Restore
cp -r backups/chroma_db_20260522 data/chroma_db
cp backups/audit_20260522.db data/audit.db
```

## Scaling

- **Horizontal**: Multiple Streamlit instances behind load balancer
- **Vertical**: Increase Gemini API quota for higher throughput
- **Storage**: ChromaDB scales with PersistentVolumeClaims

## Security Checklist

- [ ] Enable HTTPS/TLS
- [ ] Set strong API key permissions
- [ ] Configure network policies
- [ ] Enable audit logging
- [ ] Set up regular backups
- [ ] Implement rate limiting
- [ ] Use secrets management
