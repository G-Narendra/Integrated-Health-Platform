"""
Subsystem 4: Appointment Management (Agent + Tools)
Intelligent appointment scheduling with tool integration.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseHealthcareAgent
from src.core.knowledge_base import SharedKnowledgeBase
from src.tools.calendar import CalendarTool
from src.tools.email import EmailTool
from src.utils.logger import logger


class AppointmentSystem(BaseHealthcareAgent):
    """Intelligent appointment scheduling with tool use."""

    def __init__(self):
        super().__init__(model="gemini-2.5-flash-lite", temperature=0.1)
        self.calendar = CalendarTool()
        self.email = EmailTool()

    def system_prompt(self) -> str:
        return """You are an Appointment Management Agent in the UAE Integrated Healthcare Platform.
Your role is to:
1. Help patients book, reschedule, or cancel appointments
2. Check doctor availability
3. Send confirmation and reminder messages
4. Handle scheduling conflicts
5. Verify insurance information when provided

Available doctors and their specialties:
- DR-CARD-001: Dr. Ahmed Al Mansouri (Cardiology) - Cleveland Clinic Abu Dhabi
- DR-CARD-002: Dr. Fatima Al Hashimi (Cardiology) - Burjeel Hospital Dubai
- DR-ORTH-001: Dr. Mohammed Al Shehhi (Orthopedics) - NMC Royal Hospital
- DR-PED-001: Dr. Aisha Al Ketbi (Pediatrics) - Al Jalila Children's Hospital
- DR-GEN-001: Dr. Omar Al Shamsi (General Medicine) - HealthHub Abu Dhabi

Always confirm booking details with the patient and provide clear instructions."""

    def execute(
        self,
        request: Dict,
        user_context: Dict,
        knowledge_base: SharedKnowledgeBase,
    ) -> Dict[str, Any]:
        """Handle appointment requests using tools."""
        patient_request = request.get("query", "")
        patient_email = request.get("patient_email", request.get("email", ""))
        patient_name = request.get("patient_name", request.get("name", "Patient"))

        logger.info(f"Processing appointment request for {patient_email}")

        # Use LLM to parse the appointment intent
        parsed = self._parse_appointment_intent(patient_request)
        intent = parsed.get("intent", "unknown")
        doctor_id = parsed.get("doctor_id", "")
        preferred_date = parsed.get("date", "")
        preferred_time = parsed.get("time", "")

        actions_taken = []

        if intent == "book":
            # Check availability
            if doctor_id and preferred_date:
                slots = self.calendar.check_doctor_availability(doctor_id, preferred_date)
                if slots and (not preferred_time or preferred_time in slots):
                    # Book the appointment
                    booking = self.calendar.book_appointment(
                        patient_name=patient_name,
                        patient_email=patient_email,
                        doctor_id=doctor_id,
                        date=preferred_date,
                        time=preferred_time or slots[0],
                        reason=parsed.get("reason", ""),
                    )
                    actions_taken.append(f"Booked appointment: {booking['id']}")

                    # Send confirmation
                    self.email.send_appointment_reminder(
                        patient_email=patient_email,
                        patient_name=patient_name,
                        date=preferred_date,
                        time=preferred_time or slots[0],
                        doctor=doctor_id,
                    )
                    actions_taken.append("Sent confirmation email")

                    return {
                        "appointment_result": booking,
                        "actions_taken": actions_taken,
                        "confirmation_sent": True,
                        "available_slots": slots[:5],
                        "subsystem": "appointment",
                    }
                else:
                    return {
                        "appointment_result": None,
                        "actions_taken": ["Checked availability - no matching slots"],
                        "available_slots": slots[:5] if slots else [],
                        "message": "No available slots found. Please try another date or doctor.",
                        "subsystem": "appointment",
                    }

        elif intent == "check":
            slots = self.calendar.check_doctor_availability(
                doctor_id or parsed.get("specialty", ""), preferred_date
            )
            return {
                "appointment_result": None,
                "actions_taken": ["Checked availability"],
                "available_slots": slots[:10],
                "message": f"Available slots for {doctor_id} on {preferred_date}.",
                "subsystem": "appointment",
            }

        elif intent == "cancel":
            appointment_id = parsed.get("appointment_id", "")
            cancelled = self.calendar.cancel_appointment(appointment_id)
            return {
                "appointment_result": {"cancelled": cancelled, "id": appointment_id},
                "actions_taken": [f"Cancelled appointment {appointment_id}"],
                "confirmation_sent": False,
                "subsystem": "appointment",
            }

        return {
            "appointment_result": None,
            "actions_taken": ["Could not parse appointment intent"],
            "message": "I couldn't understand the appointment request. Please specify what you'd like to do (book, check, or cancel).",
            "subsystem": "appointment",
        }

    def _parse_appointment_intent(self, text: str) -> Dict:
        """Parse appointment request to extract intent and details."""
        prompt = f"""Parse this appointment request and extract key details.

Request: {text}

Return JSON:
{{
    "intent": "book|check|cancel|unknown",
    "doctor_id": "doctor ID or empty string",
    "specialty": "specialty or empty string",
    "date": "YYYY-MM-DD or empty string",
    "time": "HH:MM or empty string",
    "patient_name": "name or empty string",
    "reason": "reason or empty string",
    "appointment_id": "ID for cancellations or empty string"
}}"""
        return self.generate_structured(prompt)
