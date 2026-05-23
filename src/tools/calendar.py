"""
Calendar and appointment scheduling tool.
"""

from datetime import datetime, timedelta
from typing import Dict, List


class CalendarTool:
    """Tool for managing appointments and schedules."""

    def __init__(self):
        self._appointments: List[Dict] = []
        self._doctor_schedules: Dict[str, List[str]] = {}

    def check_doctor_availability(
        self, doctor_id: str, date: str, specialty: str = ""
    ) -> List[str]:
        """Get available time slots for a doctor on a given date."""
        if doctor_id not in self._doctor_schedules:
            # Default schedule: 9 AM to 5 PM, 30-min slots
            slots = []
            for hour in range(9, 17):
                slots.append(f"{hour:02d}:00")
                slots.append(f"{hour:02d}:30")
            self._doctor_schedules[doctor_id] = slots

        # Remove booked slots
        booked = [
            a["time"]
            for a in self._appointments
            if a["doctor_id"] == doctor_id and a["date"] == date
        ]
        return [s for s in self._doctor_schedules[doctor_id] if s not in booked]

    def book_appointment(
        self,
        patient_name: str,
        patient_email: str,
        doctor_id: str,
        date: str,
        time: str,
        reason: str = "",
    ) -> Dict:
        """Book an appointment."""
        appointment = {
            "id": f"APT-{len(self._appointments) + 1:04d}",
            "patient_name": patient_name,
            "patient_email": patient_email,
            "doctor_id": doctor_id,
            "date": date,
            "time": time,
            "reason": reason,
            "status": "confirmed",
            "created_at": datetime.now().isoformat(),
        }
        self._appointments.append(appointment)
        return appointment

    def get_upcoming_appointments(self, patient_email: str = None) -> List[Dict]:
        """Get upcoming appointments."""
        today = datetime.now().date().isoformat()
        result = [
            a for a in self._appointments
            if a["date"] >= today
        ]
        if patient_email:
            result = [a for a in result if a["patient_email"] == patient_email]
        return sorted(result, key=lambda x: (x["date"], x["time"]))

    def cancel_appointment(self, appointment_id: str) -> bool:
        """Cancel an appointment."""
        for a in self._appointments:
            if a["id"] == appointment_id:
                a["status"] = "cancelled"
                return True
        return False
