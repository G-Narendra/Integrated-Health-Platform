"""
Pydantic models for API responses.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class HealthcareResponse(BaseModel):
    """Standard healthcare API response."""
    success: bool = Field(..., description="Whether the request was successful")
    subsystem_used: str = Field(..., description="Which subsystem handled the request")
    result: Optional[Dict[str, Any]] = Field(None, description="The subsystem's response")
    audit_id: Optional[str] = Field(None, description="Audit trail identifier")
    duration_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class SystemStatus(BaseModel):
    """System health status response."""
    status: str = Field(..., description="Overall system status")
    subsystems: Dict[str, Any] = Field(..., description="Status of each subsystem")
    knowledge_base: Dict[str, int] = Field(..., description="Knowledge base document counts")


class AuditLogResponse(BaseModel):
    """Audit log query response."""
    audit_logs: List[Dict[str, Any]] = Field(..., description="List of audit log entries")
    count: int = Field(..., description="Number of entries returned")
