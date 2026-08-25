"""
FastAPI backend for the Integrated Healthcare Platform.
Provides RESTful API endpoints for all healthcare subsystems.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional

from src.api.routes.health import router as health_router
from src.api.routes.chat import router as chat_router
from src.api.routes.documents import router as documents_router
from src.core.healthcare_orchestrator import HealthcareOrchestrator

app = FastAPI(
    title="Integrated Healthcare Platform API",
    version="1.0.0",
    description="AI-powered healthcare platform for UAE - RAG + Multi-Agent + Fine-Tuning + Human-in-Loop",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware — restrict to known origins in production.
import os
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.include_router(documents_router, prefix="/api/v1", tags=["Documents"])


@app.on_event("startup")
async def startup():
    """Initialize the orchestrator on startup."""
    orchestrator = HealthcareOrchestrator()


@app.get("/")
async def root():
    """Root endpoint - redirects to API docs."""
    return {
        "message": "Integrated Healthcare Platform API",
        "version": "1.0.0",
        "docs": "/api/docs",
    }
