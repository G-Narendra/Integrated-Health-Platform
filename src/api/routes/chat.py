"""
Chat and healthcare request endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

from src.core.healthcare_orchestrator import HealthcareOrchestrator
from src.api.schemas.request import HealthcareRequest, UserContext

router = APIRouter()
orchestrator = HealthcareOrchestrator()


@router.post("/request")
async def handle_request(request: HealthcareRequest):
    """
    Main API endpoint - routes to appropriate subsystem.
    """
    try:
        user_context = {
            "user_id": request.user_id or "API-USER",
            "role": request.user_role or "system",
            "facility": request.facility or "UAE Integrated Healthcare Platform",
        }

        result = orchestrator.handle_request(
            request=request.dict(exclude_none=True),
            user_context=user_context,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diagnosis")
async def clinical_diagnosis(
    query: str,
    patient_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """Dedicated endpoint for clinical diagnosis."""
    request = {
        "request_type": "clinical_diagnosis",
        "query": query,
        "patient_id": patient_id or "UNKNOWN",
    }
    user_context = {
        "user_id": user_id or "API-USER",
        "role": "physician",
        "facility": "Cleveland Clinic Abu Dhabi",
    }
    return orchestrator.handle_request(request, user_context)


@router.post("/prescription/verify")
async def verify_prescription(
    query: str,
    patient_profile: Optional[Dict] = None,
    user_id: Optional[str] = None,
):
    """Dedicated endpoint for prescription verification."""
    request = {
        "request_type": "prescription_verify",
        "query": query,
        "patient_profile": patient_profile or {},
    }
    user_context = {
        "user_id": user_id or "API-USER",
        "role": "pharmacist",
        "facility": "Burjeel Hospital Dubai",
    }
    return orchestrator.handle_request(request, user_context)
