# 🏥 Integrated Healthcare Platform

**RAG + Multi-Agent + Fine-Tuning + Human-in-Loop (Combination)**

An AI-powered healthcare platform for UAE medical facilities that combines five advanced AI techniques into a cohesive, production-ready system.

## ✨ Features

| Subsystem | Technique | Description |
|-----------|-----------|-------------|
| 🔍 **Clinical Diagnosis** | Multi-Agent | Differential diagnosis with evidence-based guidelines |
| 💊 **Prescription Verification** | RAG + Human-in-Loop | Drug safety checks with pharmacist review |
| 📋 **Medical Records** | Fine-Tuned | Bilingual (Arabic/English) record processing |
| 📅 **Appointment Management** | Agent + Tools | Intelligent scheduling with automated reminders |
| 🔬 **Medical Research** | Agentic RAG | Evidence-based research with UAE-specific literature |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key ([Get one free](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Clone the repository
cd 09_integrated_healthcare_platform

# Activate common virtual environment
source ../venv/bin/activate  # Linux/Mac
# OR
..\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY="your_key_here"  # Linux/Mac
# OR
set GEMINI_API_KEY=your_key_here  # Windows

# Launch the app
streamlit run app.py
```

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v
```

## 📁 Project Structure

```
09_integrated_healthcare_platform/
├── app.py                    # Main Streamlit application
├── Dockerfile                # Container configuration
├── docker-compose.yml        # Multi-service orchestration
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files
├── data/                     # Data and vector database
├── docs/                     # Documentation
├── src/                      # Source code modules
├── scripts/                  # Utility scripts
├── tests/                    # Test suites
└── deployment/               # Kubernetes, monitoring, Terraform
```

## 🔐 Compliance

- **UAE MOH Compliant** - Follows Ministry of Health guidelines
- **HIPAA-ready** - Audit logging, data encryption patterns
- **Human-in-Loop** - All clinical decisions require licensed professional review
- **Audit Trail** - Every AI interaction logged with timestamp and hash

## 📊 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| AI/LLM | Google Gemini 2.0 Flash |
| Embeddings | Google Text-Embedding-004 |
| Vector DB | ChromaDB |
| Database | SQLite3 |
| Containerization | Docker + Docker Compose |

---

*Built for UAE AI Student Projects - Capstone Project 9*
