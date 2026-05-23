"""
Central orchestration layer for the Integrated Healthcare Platform.
Routes requests to appropriate subsystems and manages the shared knowledge base.
"""

import time
from enum import Enum
from typing import Any, Dict, Optional

from src.core.audit_logger import ComplianceAuditLogger
from src.core.knowledge_base import SharedKnowledgeBase
from src.subsystems.diagnosis_system import ClinicalDiagnosisSystem
from src.subsystems.prescription_system import PrescriptionVerificationSystem
from src.subsystems.records_system import MedicalRecordsSystem
from src.subsystems.appointment_system import AppointmentSystem
from src.subsystems.research_system import MedicalResearchSystem
from src.utils.logger import logger
from src.utils.monitoring import metrics


class RequestType(Enum):
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"
    PRESCRIPTION_VERIFY = "prescription_verify"
    MEDICAL_RECORD = "medical_record"
    APPOINTMENT = "appointment"
    RESEARCH = "research"
    UNKNOWN = "unknown"


class HealthcareOrchestrator:
    """
    Central orchestrator that routes healthcare requests to the appropriate subsystem.
    Manages shared knowledge base, audit logging, and cross-subsystem coordination.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, chroma_path: str = "./data/chroma_db"):
        if self._initialized:
            return
        self._initialized = True

        logger.info("Initializing Healthcare Orchestrator...")

        # Shared components
        self.knowledge_base = SharedKnowledgeBase(chroma_path=chroma_path)
        self.audit_logger = ComplianceAuditLogger()

        # Initialize subsystems
        self.subsystems = {
            RequestType.CLINICAL_DIAGNOSIS: ClinicalDiagnosisSystem(),
            RequestType.PRESCRIPTION_VERIFY: PrescriptionVerificationSystem(),
            RequestType.MEDICAL_RECORD: MedicalRecordsSystem(),
            RequestType.APPOINTMENT: AppointmentSystem(),
            RequestType.RESEARCH: MedicalResearchSystem(),
        }

        logger.info("Healthcare Orchestrator initialized successfully")

    def classify_request(self, request: Dict) -> RequestType:
        """Determine which subsystem should handle this request based on content."""
        request_text = request.get("query", request.get("text", "")).lower()
        request_type = request.get("request_type", "")

        if request_type:
            try:
                return RequestType(request_type)
            except ValueError:
                pass

        # Rule-based classification
        diagnosis_keywords = ["diagnos", "symptom", "chest pain", "fever", "cough",
                              "headache", "pain", "injury", "wound", "infection"]
        prescription_keywords = ["prescription", "medication", "drug", "dosage",
                                 "interaction", "medicine", "rx", "pharma"]
        record_keywords = ["record", "history", "past visit", "medical history",
                           "chart", "summary", "extract"]
        appointment_keywords = ["appointment", "schedule", "book", "cancel appointment",
                                "reschedule", "visit"]
        research_keywords = ["research", "latest treatment", "studies", "clinical trial",
                             "evidence", "literature", "guideline"]

        if any(kw in request_text for kw in diagnosis_keywords):
            return RequestType.CLINICAL_DIAGNOSIS
        elif any(kw in request_text for kw in prescription_keywords):
            return RequestType.PRESCRIPTION_VERIFY
        elif any(kw in request_text for kw in record_keywords):
            return RequestType.MEDICAL_RECORD
        elif any(kw in request_text for kw in appointment_keywords):
            return RequestType.APPOINTMENT
        elif any(kw in request_text for kw in research_keywords):
            return RequestType.RESEARCH

        return RequestType.UNKNOWN

    def handle_request(
        self,
        request: Dict,
        user_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point - routes request to appropriate subsystem.

        Args:
            request: {
                "query": "Patient has chest pain...",
                "patient_id": "PT-12345",
                "request_type": "clinical_diagnosis" (optional)
            }
            user_context: {
                "user_id": "DR-001",
                "role": "physician",
                "facility": "Cleveland Clinic Abu Dhabi"
            }

        Returns:
            {
                "success": True/False,
                "subsystem_used": "...",
                "result": {...},
                "audit_id": "AUD-2026..."
            }
        """
        user_context = user_context or {
            "user_id": "SYSTEM",
            "role": "system",
            "facility": "UAE Integrated Healthcare Platform",
        }

        # Classify request
        if "request_type" in request and request["request_type"]:
            try:
                request_type = RequestType(request["request_type"])
            except ValueError:
                request_type = self.classify_request(request)
        else:
            request_type = self.classify_request(request)

        if request_type == RequestType.UNKNOWN:
            return {
                "success": False,
                "subsystem_used": "unknown",
                "result": {"error": "Could not classify request. Please be more specific."},
                "audit_id": None,
            }

        # Route to appropriate subsystem
        subsystem = self.subsystems.get(request_type)
        if subsystem is None:
            return {
                "success": False,
                "result": {"error": f"No subsystem available for {request_type.value}"},
                "audit_id": None,
            }

        # Execute with timing and audit logging
        start_time = time.time()
        try:
            result = subsystem.execute(
                request=request,
                user_context=user_context,
                knowledge_base=self.knowledge_base,
            )

            duration_ms = (time.time() - start_time) * 1000
            metrics.record_latency(request_type.value, duration_ms)

            # Log to audit trail
            audit_id = self.audit_logger.log_request(
                request_type=request_type.value,
                request=request,
                user_context=user_context,
                result=result,
                success=True,
                duration_ms=duration_ms,
            )

            return {
                "success": True,
                "subsystem_used": request_type.value,
                "result": result,
                "audit_id": audit_id,
                "duration_ms": round(duration_ms, 2),
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            metrics.record_error(request_type.value)
            logger.error(f"Subsystem {request_type.value} failed: {e}")

            audit_id = self.audit_logger.log_request(
                request_type=request_type.value,
                request=request,
                user_context=user_context,
                result={"error": str(e)},
                success=False,
                duration_ms=duration_ms,
            )

            return {
                "success": False,
                "subsystem_used": request_type.value,
                "error": str(e),
                "audit_id": audit_id,
                "duration_ms": round(duration_ms, 2),
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        kb_status = self.knowledge_base.status()
        subsystem_names = {
            "clinical_diagnosis": "Clinical Diagnosis (Multi-Agent)",
            "prescription_verify": "Prescription Verification (RAG + Human-in-Loop)",
            "medical_record": "Medical Records (Bilingual Processing)",
            "appointment": "Appointment Management (Agent + Tools)",
            "research": "Medical Research (Agentic RAG)",
        }

        return {
            "system": "Integrated Healthcare Platform",
            "version": "1.0.0",
            "status": "operational",
            "subsystems": {
                name: {"name": display_name, "status": "operational"}
                for name, display_name in subsystem_names.items()
            },
            "knowledge_base": kb_status,
            "audit_log_count": len(self.audit_logger.get_recent_logs(1)),
            "uptime_seconds": metrics.get_uptime_seconds(),
        }
