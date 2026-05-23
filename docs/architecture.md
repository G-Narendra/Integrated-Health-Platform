# Integrated Healthcare Platform - Architecture

## Overview

The Integrated Healthcare Platform combines five AI techniques (Multi-Agent, RAG, Fine-Tuning, Agent with Tools, Agentic RAG) into a cohesive production system for UAE healthcare facilities.

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND (app.py)                      │
│  ├─ Tab 1: Clinical Diagnosis                                      │
│  ├─ Tab 2: Prescription Verification                                │
│  ├─ Tab 3: Medical Records Processing                              │
│  ├─ Tab 4: Appointment Management                                   │
│  └─ Tab 5: Medical Research                                        │
└─────────────────┬──────────────────────────────────────────────────┘
                  │
                  v
┌────────────────────────────────────────────────────────────────────┐
│                   HEALTHCARE ORCHESTRATOR                           │
│  Routes requests to appropriate subsystems                         │
└─────────────────┬──────────────────────────────────────────────────┘
                  │
        ┌─────────┼─────────┬──────────┬──────────┬─────────┐
        │         │         │          │          │         │
        v         v         v          v          v         v
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Clinical │ │   Rx     │ │ Medical  │ │ Appoint- │ │ Medical  │
│ Diagnosis│ │ Verify   │ │ Records  │ │ ment     │ │ Research │
│(Multi-   │ │(RAG +    │ │(Fine-    │ │(Agent +  │ │(Agentic  │
│ Agent)   │ │ Human)   │ │ Tuned)   │ │ Tools)   │ │ RAG)     │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
     │            │            │            │            │
     └────────────┴────────────┴────────────┴────────────┘
                  │
                  v
┌────────────────────────────────────────────────────────────────────┐
│                     SHARED INFRASTRUCTURE                          │
│  ├─ Gemini 2.0 Flash (LLM)                                         │
│  ├─ ChromaDB (Vector Store)                                        │
│  ├─ SQLite (Audit Logs + Appointments)                              │
│  └─ Embedding API (Text-Embedding-004)                              │
└────────────────────────────────────────────────────────────────────┘
```

## Subsystem Details

### 1. Clinical Diagnosis (Multi-Agent)
- **Technique**: Multi-Agent using LangGraph-like orchestration
- **LLM**: Gemini 2.0 Flash with medical prompting
- **RAG**: Clinical guidelines from ChromaDB
- **Output**: Differential diagnosis, recommended tests, urgency level
- **Human-in-Loop**: Requires physician review

### 2. Prescription Verification (RAG + Human-in-Loop)
- **Technique**: RAG with drug database + pharmacist review
- **LLM**: Gemini 2.0 Flash with pharmacology prompting
- **RAG**: Drug formulary from ChromaDB (20+ common UAE drugs)
- **Output**: Safety check, dose verification, interaction check
- **Human-in-Loop**: Flags for pharmacist review when issues detected

### 3. Medical Records (Fine-Tuned)
- **Technique**: Fine-tuned bilingual processing
- **LLM**: Gemini 2.0 Flash with medical terminology focus
- **Capabilities**: Summarize, extract structured data, translate (Arabic/English)
- **Output**: Structured clinical notes with preserved medical terminology

### 4. Appointment Management (Agent + Tools)
- **Technique**: Agent with tool use
- **LLM**: Gemini 2.0 Flash for task planning
- **Tools**: Calendar scheduling, database CRUD, notifications
- **Output**: Confirmed appointments with audit trail
- **Storage**: SQLite database

### 5. Medical Research (Agentic RAG)
- **Technique**: Agentic RAG with iterative research
- **LLM**: Gemini 2.0 Flash
- **RAG**: Medical literature in ChromaDB (10+ UAE research topics)
- **Output**: Evidence-based research reports with quality ratings

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit 1.40+ |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google Text-Embedding-004 |
| Vector Store | ChromaDB (PersistentClient) |
| Database | SQLite3 |
| Audit | SQLite3 with SHA-256 hashing |
| Auth | Basic token (JWT in production) |
| Deployment | Docker + Docker Compose |

## Data Flow

1. User interacts with Streamlit UI
2. Request is processed by subsystem-specific handler
3. Subsystem queries LLM (Gemini) with specialized prompting
4. RAG subsystems retrieve context from ChromaDB
5. Results are logged to SQLite audit trail
6. Response displayed to user with appropriate disclaimers

## Security & Compliance

- **Data Protection**: All patient data minimally exposed; IDs are hashed
- **Audit Trail**: Every AI interaction logged with timestamp
- **Human Oversight**: All clinical decisions require licensed professional review
- **MOH Compliance**: Outputs formatted per UAE Ministry of Health guidelines
- **HIPAA Readiness**: Encryption-ready, access control patterns implemented

## Performance

- **Response Time**: <5 seconds for most operations
- **Concurrent Users**: Limited by Gemini API quota
- **Storage**: ChromaDB persistent + SQLite for audit
- **Scalability**: Horizontal scaling via Docker
