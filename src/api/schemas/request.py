"""
Pydantic models for API requests.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class HealthcareRequest(BaseModel):
    """Main healthcare request model."""
    query: str = Field(..., description="The user's query or request text")
    patient_id: Optional[str] = Field(None, description="Patient ID if applicable")
    patient_name: Optional[str] = Field(None, description="Patient name")
    patient_email: Optional[str] = Field(None, description="Patient email for appointments")
    request_type: Optional[str] = Field(None, description="Force a specific subsystem")
    user_id: Optional[str] = Field(None, description="User/requester ID")
    user_role: Optional[str] = Field(None, description="User role (physician, pharmacist, admin, patient)")
    facility: Optional[str] = Field(None, description="Healthcare facility name")
    task: Optional[str] = Field("summarize", description="Task type for records subsystem")
    patient_profile: Optional[Dict[str, Any]] = Field(None, description="Patient profile for prescription verification")
    target_language: Optional[str] = Field(None, description="Target language for translation (en/ar)")


class UserContext(BaseModel):
    """User context for authentication and authorization."""
    user_id: str = Field(..., description="User identifier")
    role: str = Field(..., description="User role: physician, pharmacist, nurse, admin, patient")
    facility: str = Field(..., description="Healthcare facility name")
