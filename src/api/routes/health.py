"""
Health check and system status endpoints.
"""

from fastapi import APIRouter
from src.core.healthcare_orchestrator import HealthcareOrchestrator
from src.utils.monitoring import metrics

router = APIRouter()
orchestrator = HealthcareOrchestrator()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Integrated Healthcare Platform",
        "version": "1.0.0",
    }


@router.get("/system/status")
async def system_status():
    """Get comprehensive system status."""
    try:
        status = orchestrator.get_system_status()
        stats = metrics.get_all_stats()
        return {**status, "performance": stats}
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
        }


@router.get("/system/audit")
async def audit_logs(limit: int = 10):
    """Get recent audit logs."""
    logs = orchestrator.audit_logger.get_recent_logs(limit)
    return {"audit_logs": logs, "count": len(logs)}
