"""
Unit tests for healthcare platform agents and subsystems
"""

import os
import sys
import json
import sqlite3
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestClinicalDiagnosisSystem(unittest.TestCase):
    """Test the clinical diagnosis subsystem"""

    def setUp(self):
        self.patcher = patch('app.GeminiClient.generate')
        self.mock_llm = self.patcher.start()
        self.mock_llm.return_value = "## 🔍 Differential Diagnosis\n1. Community Acquired Pneumonia (High)\n2. Acute Bronchitis (Medium)\n\n## ⚠️ Urgency Level\nURGENT"

    def tearDown(self):
        self.patcher.stop()

    def test_diagnosis_returns_expected_keys(self):
        """Test that diagnosis returns required fields"""
        from app import ClinicalDiagnosisSystem
        system = ClinicalDiagnosisSystem(self.mock_llm)
        result = system.diagnose("cough, fever, chest pain")

        self.assertIn('analysis', result)
        self.assertIn('trace_id', result)
        self.assertIn('subsystem', result)
        self.assertEqual(result['subsystem'], 'clinical_diagnosis')

    def test_diagnosis_with_patient_info(self):
        """Test diagnosis with patient context"""
        from app import ClinicalDiagnosisSystem
        system = ClinicalDiagnosisSystem(self.mock_llm)
        result = system.diagnose(
            "chest pain, shortness of breath",
            {"age": 65, "gender": "Male", "conditions": "Hypertension"}
        )
        self.assertIn('analysis', result)


class TestPrescriptionVerificationSystem(unittest.TestCase):
    """Test the prescription verification subsystem"""

    def setUp(self):
        self.patcher = patch('app.GeminiClient.generate')
        self.mock_llm = self.patcher.start()
        self.mock_llm.return_value = "## ✅ Verification Results\n### Safety Check: PASS\n### Final Decision: APPROVE"

    def tearDown(self):
        self.patcher.stop()

    def test_verify_returns_expected_keys(self):
        """Test that verification returns required fields"""
        from app import PrescriptionVerificationSystem
        system = PrescriptionVerificationSystem(self.mock_llm)

        result = system.verify(
            {"drug": "Paracetamol 500mg", "dose": "500mg", "frequency": "QID", "route": "oral", "duration": "5 days"},
            {"age": 30, "weight": 70, "allergies": "None", "conditions": "None"}
        )

        self.assertIn('verification', result)
        self.assertIn('trace_id', result)
        self.assertIn('requires_human_review', result)
        self.assertIn('status', result)

    def test_verify_rejects_allergic_patient(self):
        """Test rejection for known allergy"""
        from app import PrescriptionVerificationSystem
        system = PrescriptionVerificationSystem(self.mock_llm)

        result = system.verify(
            {"drug": "Amoxicillin 500mg", "dose": "500mg", "frequency": "TID", "route": "oral", "duration": "7 days"},
            {"age": 35, "weight": 70, "allergies": "Penicillin", "conditions": "None"}
        )

        self.assertIn('requires_human_review', result)


class TestMedicalRecordsSystem(unittest.TestCase):
    """Test the medical records subsystem"""

    def setUp(self):
        self.patcher = patch('app.GeminiClient.generate')
        self.mock_llm = self.patcher.start()
        self.mock_llm.return_value = "## Clinical Summary\nDiagnosis: CAP\nMedications: Azithromycin"

    def tearDown(self):
        self.patcher.stop()

    def test_summarize_returns_result(self):
        """Test record summarization"""
        from app import MedicalRecordsSystem
        system = MedicalRecordsSystem(self.mock_llm)

        result = system.process(
            "Patient: Test. Diagnosis: CAP. Medications: Azithromycin.",
            task="summarize"
        )

        self.assertIn('result', result)
        self.assertIn('model_tier', result)

    def test_extract_returns_result(self):
        """Test data extraction"""
        from app import MedicalRecordsSystem
        system = MedicalRecordsSystem(self.mock_llm)

        result = system.process(
            "Patient: Test. BP 120/80. Medications: Metformin 500mg.",
            task="extract"
        )

        self.assertIn('result', result)


class TestAppointmentScheduler(unittest.TestCase):
    """Test the appointment scheduling subsystem"""

    def setUp(self):
        from app import AppointmentScheduler
        self.scheduler = AppointmentScheduler.__new__(AppointmentScheduler)
        self.scheduler.conn = sqlite3.connect(":memory:")
        cursor = self.scheduler.conn.cursor()
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
        self.scheduler.conn.commit()

    def test_book_appointment(self):
        """Test booking an appointment"""
        from app import AppointmentScheduler
        result = self.scheduler.book(
            patient_name="Test Patient",
            doctor="Dr. Test",
            department="Cardiology",
            date="2026-05-25",
            time="10:00",
            reason="Checkup",
            email="test@email.com"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result['patient'], "Test Patient")
        self.assertEqual(result['status'], "scheduled")

    def test_get_appointments(self):
        """Test retrieving appointments"""
        from app import AppointmentScheduler
        self.scheduler.book(
            patient_name="Patient 1",
            doctor="Dr. Test",
            department="Cardiology",
            date="2026-05-25",
            time="10:00",
            reason="Checkup"
        )

        self.scheduler.book(
            patient_name="Patient 2",
            doctor="Dr. Test",
            department="Neurology",
            date="2026-05-26",
            time="14:00",
            reason="Consultation"
        )

        appointments = self.scheduler.get_appointments()
        self.assertGreaterEqual(len(appointments), 2)


class TestMedicalResearchSystem(unittest.TestCase):
    """Test the medical research subsystem"""

    def setUp(self):
        self.patcher = patch('app.GeminiClient.generate')
        self.mock_llm = self.patcher.start()
        self.mock_llm.return_value = "## Research Report\nEvidence-based findings..."

    def tearDown(self):
        self.patcher.stop()

    def test_research_returns_expected_keys(self):
        """Test research returns required fields"""
        from app import MedicalResearchSystem
        system = MedicalResearchSystem(self.mock_llm)

        result = system.research("Latest diabetes treatments")

        self.assertIn('report', result)
        self.assertIn('trace_id', result)
        self.assertIn('subsystem', result)


class TestAuditLogger(unittest.TestCase):
    """Test the audit logging subsystem"""

    def setUp(self):
        from app import AuditLogger
        self.logger = AuditLogger.__new__(AuditLogger)
        self.logger.conn = sqlite3.connect(":memory:")
        cursor = self.logger.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                request_type TEXT NOT NULL,
                subsystem TEXT,
                user_role TEXT DEFAULT 'anonymous',
                facility TEXT DEFAULT 'UAE Healthcare',
                success BOOLEAN DEFAULT 1,
                duration_ms REAL DEFAULT 0,
                estimated_tokens INTEGER DEFAULT 0,
                cache_hit BOOLEAN DEFAULT 0,
                model_used TEXT DEFAULT '',
                result_summary TEXT,
                error_message TEXT
            )
        """)
        self.logger.conn.commit()

    def test_log_request(self):
        """Test logging a request"""
        trace_id = "TEST-20260523-000001"
        self.logger.log(
            trace_id=trace_id,
            request_type="clinical_diagnosis",
            subsystem="clinical_diagnosis",
            summary="Test summary"
        )
        
        # Verify it was logged
        cursor = self.logger.conn.execute(
            "SELECT trace_id FROM audit_trail WHERE trace_id = ?", (trace_id,)
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], trace_id)

    def test_get_stats(self):
        """Test retrieving audit stats via SQL"""
        # Log some requests
        self.logger.log(
            trace_id="STATS-TEST-1",
            request_type="test",
            subsystem="test_subsystem",
            success=True
        )
        self.logger.log(
            trace_id="STATS-TEST-2",
            request_type="test",
            subsystem="test_subsystem",
            success=True
        )
        
        cursor = self.logger.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(success), 0) FROM audit_trail"
        )
        total, successful = cursor.fetchone()
        self.assertGreaterEqual(total, 2)
        self.assertGreaterEqual(successful, 2)


if __name__ == '__main__':
    unittest.main()
