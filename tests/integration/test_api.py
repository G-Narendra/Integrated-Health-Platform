"""
Integration tests for the healthcare platform
Tests the end-to-end flow of all subsystems
"""

import os
import sys
import json
import sqlite3
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestHealthcareOrchestrator(unittest.TestCase):
    """Test the central healthcare orchestrator"""

    def setUp(self):
        self.patcher = patch('app.GeminiClient.generate')
        self.mock_llm = self.patcher.start()
        self.mock_llm.return_value = "Test response from AI assistant"

    def tearDown(self):
        self.patcher.stop()

    def test_full_diagnosis_flow(self):
        """Test end-to-end diagnosis flow"""
        from app import ClinicalDiagnosisSystem
        system = ClinicalDiagnosisSystem(self.mock_llm)

        result = system.diagnose(
            "68-year-old male with sudden onset chest pain, radiating to left arm, SOB, diaphoresis. BP 160/95, HR 110.",
            {"age": 68, "gender": "Male", "conditions": "Hypertension, Type 2 Diabetes, Smoker"}
        )

        self.assertIsNotNone(result)
        self.assertIn('analysis', result)
        self.assertIn('subsystem', result)
        self.assertEqual(result['subsystem'], 'clinical_diagnosis')

    def test_full_prescription_flow(self):
        """Test end-to-end prescription verification"""
        from app import PrescriptionVerificationSystem
        system = PrescriptionVerificationSystem(self.mock_llm)

        result = system.verify(
            {"drug": "Metformin 500mg", "dose": "500mg", "frequency": "BID", "route": "oral", "duration": "30 days"},
            {"age": 55, "weight": 80, "allergies": "None", "conditions": "Type 2 Diabetes"}
        )

        self.assertIsNotNone(result)
        self.assertIn('verification', result)
        self.assertIn('status', result)

    def test_full_appointment_flow(self):
        """Test end-to-end appointment booking"""
        from app import AppointmentScheduler
        scheduler = AppointmentScheduler.__new__(AppointmentScheduler)
        scheduler.conn = sqlite3.connect(":memory:")
        cursor = scheduler.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                patient_email TEXT,
                doctor TEXT NOT NULL,
                department TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'scheduled',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        scheduler.conn.commit()

        # Book appointment
        result = scheduler.book(
            patient_name="Ahmed Hassan",
            doctor="Dr. Sarah Hassan (Cardiology)",
            department="Cardiology",
            date="2026-06-01",
            time="09:00",
            reason="Annual cardiac checkup",
            email="ahmed@email.com"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result['patient'], "Ahmed Hassan")

        # Verify it appears in list
        appointments = scheduler.get_appointments()
        self.assertGreaterEqual(len(appointments), 1)
        self.assertEqual(appointments[0]['patient_name'], "Ahmed Hassan")


if __name__ == '__main__':
    unittest.main()
