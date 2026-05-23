# API Reference - Integrated Healthcare Platform

## Overview

The platform provides RESTful and Streamlit-based APIs for all healthcare subsystems. This document covers the internal APIs used by the Streamlit frontend and the FastAPI endpoints.

## FastAPI Endpoints

### POST /api/v1/request

Main entry point - routes to appropriate subsystem.

**Request Body:**
```json
{
  "query": "Patient has chest pain and shortness of breath",
  "patient_id": "PT-12345",
  "request_type": "clinical_diagnosis",
  "patient_case": {},
  "prescription": {},
  "patient_profile": {}
}
```

**Response:**
```json
{
  "success": true,
  "subsystem_used": "clinical_diagnosis",
  "result": {},
  "audit_id": "AUD-20260518-000001"
}
```

### POST /api/v1/diagnosis

Dedicated endpoint for clinical diagnosis.

**Authentication:** Role-based access (physician, nurse)

**Request Body:**
```json
{
  "symptoms": "65-year-old male with chest pain...",
  "patient_info": {
    "age": 65,
    "gender": "Male",
    "conditions": "Hypertension, Diabetes"
  }
}
```

### POST /api/v1/prescription/verify

Dedicated endpoint for prescription verification.

**Authentication:** Pharmacist role required

**Request Body:**
```json
{
  "prescription": {
    "drug": "Amoxicillin 500mg",
    "dose": "500mg",
    "frequency": "TID",
    "route": "oral",
    "duration": "7 days"
  },
  "patient_profile": {
    "age": 35,
    "weight": 70,
    "allergies": "None",
    "conditions": "None"
  }
}
```

### GET /api/v1/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "subsystems": {
    "diagnosis": "operational",
    "prescription": "operational",
    "records": "operational",
    "appointments": "operational",
    "research": "operational"
  }
}
```

## Internal Python APIs

### HealthcareOrchestrator

```python
orchestrator = HealthcareOrchestrator()
result = orchestrator.handle_request(request, user_context)
```

### ClinicalDiagnosisSystem

```python
system = ClinicalDiagnosisSystem()
result = system.diagnose(symptoms, patient_info)
# Returns: { "analysis": str, "guidelines_used": [], "requires_physician_review": bool }
```

### PrescriptionVerificationSystem

```python
system = PrescriptionVerificationSystem()
result = system.verify(prescription, patient_profile)
# Returns: { "verification": str, "drug_info": str, "requires_human_review": bool, "status": str }
```

### MedicalRecordsSystem

```python
system = MedicalRecordsSystem()
result = system.process(record_text, task="summarize", language="en")
# Returns: { "result": str, "task": str, "language": str }
```

### AppointmentScheduler

```python
scheduler = AppointmentScheduler()
result = scheduler.book(patient_name, doctor, department, date, time, reason)
# Returns: { "id": int, "patient": str, ... }
appointments = scheduler.get_appointments(date="2026-05-22")
```

### MedicalResearchSystem

```python
system = MedicalResearchSystem()
result = system.research(query)
# Returns: { "report": str, "sources": [], "evidence_level": str }
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Missing required parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 403 | Forbidden - Insufficient role permissions |
| 500 | Internal Server Error - LLM or database failure |
| 503 | Service Unavailable - ChromaDB or API not responding |

## Rate Limiting

- Free tier: 60 requests/minute
- Professional: 300 requests/minute
- Enterprise: Custom limits

## Authentication

Currently uses API key in header:
```
Authorization: Bearer <gemini_api_key>
```

In production, implement OAuth2 with JWT tokens and role-based access control.
