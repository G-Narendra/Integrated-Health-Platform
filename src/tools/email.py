"""
Email notification tool for appointment reminders and alerts.
"""

from typing import Dict, List


class EmailTool:
    """Tool for sending email notifications (simulated)."""

    def __init__(self):
        self._sent_emails: List[Dict] = []

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: List[str] = None,
    ) -> Dict:
        """Send an email (simulated)."""
        email = {
            "to": to,
            "cc": cc or [],
            "subject": subject,
            "body": body,
            "status": "sent",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

        if __import__("os").getenv("DEBUG", "false").lower() == "true":
            print(f"\n[EMAIL SIMULATION]")
            print(f"  To: {to}")
            print(f"  Subject: {subject}")
            print(f"  Body: {body[:200]}...")

        self._sent_emails.append(email)
        return email

    def send_appointment_reminder(
        self, patient_email: str, patient_name: str, date: str, time: str, doctor: str
    ) -> Dict:
        """Send an appointment reminder email."""
        subject = "Appointment Reminder - UAE Integrated Healthcare"
        body = (
            f"Dear {patient_name},\n\n"
            f"This is a reminder of your upcoming appointment:\n"
            f"Date: {date}\n"
            f"Time: {time}\n"
            f"Doctor: {doctor}\n\n"
            f"Please arrive 15 minutes early. For cancellations, please notify us 24 hours in advance.\n\n"
            f"Thank you,\nUAE Integrated Healthcare Platform"
        )
        return self.send_email(patient_email, subject, body)

    def send_prescription_alert(
        self, pharmacist_email: str, prescription_details: str
    ) -> Dict:
        """Send a prescription verification alert."""
        subject = "Prescription Verification Alert - Action Required"
        return self.send_email(pharmacist_email, subject, prescription_details)

    def get_sent_emails(self, limit: int = 10) -> List[Dict]:
        """Get recently sent emails."""
        return self._sent_emails[-limit:]
