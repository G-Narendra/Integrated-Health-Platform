"""
Document processing endpoints for medical records.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from src.core.healthcare_orchestrator import HealthcareOrchestrator

router = APIRouter()
orchestrator = HealthcareOrchestrator()


@router.post("/documents/process")
async def process_document(
    file: UploadFile = File(...),
    task: str = Form("summarize"),
    user_id: str = Form("API-USER"),
):
    """
    Process a medical document (summarize, extract, or translate).
    """
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")

        request = {
            "request_type": "medical_record",
            "query": text,
            "task": task,
        }
        user_context = {
            "user_id": user_id,
            "role": "physician",
            "facility": "UAE Healthcare",
        }

        result = orchestrator.handle_request(request, user_context)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/translate")
async def translate_document(
    file: UploadFile = File(...),
    target_language: str = Form("en"),
):
    """
    Translate a medical document between Arabic and English.
    """
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")

        request = {
            "request_type": "medical_record",
            "query": text,
            "task": "translate",
            "target_language": target_language,
        }
        user_context = {
            "user_id": "API-USER",
            "role": "physician",
            "facility": "UAE Healthcare",
        }

        result = orchestrator.handle_request(request, user_context)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
