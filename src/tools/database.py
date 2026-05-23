"""
Database tool for managing patient records and operational data.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatabaseTool:
    """Tool for database operations on patient records, doctors, and facilities."""

    def __init__(self, db_path: str = "./data/healthcare.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS doctors (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                specialty TEXT NOT NULL,
                facility TEXT,
                license_number TEXT,
                is_available INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS facilities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT,
                type TEXT,
                contact TEXT
            );

            CREATE TABLE IF NOT EXISTS patient_records (
                id TEXT PRIMARY KEY,
                patient_name TEXT,
                age INTEGER,
                gender TEXT,
                contact TEXT,
                blood_group TEXT,
                allergies TEXT,
                created_at TEXT
            );
        """)
        self.conn.commit()

    def get_doctors(self, specialty: str = None) -> List[Dict]:
        """Get list of doctors, optionally filtered by specialty."""
        if specialty:
            cursor = self.conn.execute(
                "SELECT * FROM doctors WHERE specialty = ?", (specialty,)
            )
        else:
            cursor = self.conn.execute("SELECT * FROM doctors")
        return [dict(row) for row in cursor.fetchall()]

    def add_doctor(self, doctor_data: Dict) -> bool:
        """Add a doctor to the database."""
        try:
            self.conn.execute(
                """INSERT INTO doctors (id, name, specialty, facility, license_number, is_available)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    doctor_data.get("id"),
                    doctor_data.get("name"),
                    doctor_data.get("specialty"),
                    doctor_data.get("facility", ""),
                    doctor_data.get("license_number", ""),
                    1,
                ),
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def get_facilities(self) -> List[Dict]:
        """Get list of healthcare facilities."""
        cursor = self.conn.execute("SELECT * FROM facilities")
        return [dict(row) for row in cursor.fetchall()]

    def search_patients(self, query: str) -> List[Dict]:
        """Search for patients by name or ID."""
        cursor = self.conn.execute(
            "SELECT * FROM patient_records WHERE patient_name LIKE ? OR id LIKE ?",
            (f"%{query}%", f"%{query}%"),
        )
        return [dict(row) for row in cursor.fetchall()]
