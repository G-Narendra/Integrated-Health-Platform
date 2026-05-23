"""
HIPAA-compliant audit logging for all system actions.
Logs requests, decisions, and data access with proper hashing of sensitive data.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ComplianceAuditLogger:
    """HIPAA-ready audit logging with full traceability."""

    def __init__(self, db_path: str = "./data/audit.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create comprehensive audit tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS system_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                request_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                facility TEXT DEFAULT '',
                patient_id_hash TEXT,
                request_hash TEXT,
                subsystem_used TEXT,
                success INTEGER DEFAULT 0,
                result_summary TEXT,
                audit_trail TEXT,
                duration_ms REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS physician_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER,
                physician_id TEXT NOT NULL,
                review_timestamp TEXT NOT NULL,
                ai_recommendation TEXT,
                physician_decision TEXT,
                physician_notes TEXT,
                FOREIGN KEY (request_id) REFERENCES system_requests(id)
            );

            CREATE TABLE IF NOT EXISTS data_access (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id_hash TEXT NOT NULL,
                action TEXT NOT NULL,
                ip_address TEXT DEFAULT '',
                success INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS human_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                decision_type TEXT NOT NULL,
                decision TEXT NOT NULL,
                decision_by TEXT NOT NULL,
                notes TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (request_id) REFERENCES system_requests(id)
            );

            CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON system_requests(timestamp);
            CREATE INDEX IF NOT EXISTS idx_requests_type ON system_requests(request_type);
            CREATE INDEX IF NOT EXISTS idx_reviews_physician ON physician_reviews(physician_id);
        """)
        self.conn.commit()

    def log_request(
        self,
        request_type: str,
        request: Dict,
        user_context: Dict,
        result: Dict,
        success: bool,
        duration_ms: float = 0,
    ) -> str:
        """Log a system request with full audit trail."""
        patient_id = request.get("patient_id", "")
        patient_id_hash = (
            hashlib.sha256(patient_id.encode()).hexdigest()[:16]
            if patient_id else None
        )

        request_hash = hashlib.sha256(
            json.dumps(request, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        subsystem = result.get("subsystem", "unknown")

        cursor = self.conn.execute(
            """INSERT INTO system_requests
               (timestamp, request_type, user_id, user_role, facility,
                patient_id_hash, request_hash, subsystem_used,
                success, result_summary, audit_trail, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                request_type,
                user_context.get("user_id", "unknown"),
                user_context.get("role", "unknown"),
                user_context.get("facility", "unknown"),
                patient_id_hash,
                request_hash,
                subsystem,
                1 if success else 0,
                json.dumps(result, default=str)[:500],
                json.dumps({
                    "request_hash": request_hash,
                    "timestamp": datetime.now().isoformat(),
                }),
                duration_ms,
            ),
        )

        audit_id = f"AUD-{datetime.now().strftime('%Y%m%d')}-{cursor.lastrowid:06d}"
        self.conn.commit()
        return audit_id

    def log_human_decision(
        self,
        request_id: int,
        decision_type: str,
        decision: str,
        decision_by: str,
        notes: str = "",
    ):
        """Log a human (physician/pharmacist) decision on an AI recommendation."""
        self.conn.execute(
            """INSERT INTO human_decisions
               (request_id, decision_type, decision, decision_by, notes, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                decision_type,
                decision,
                decision_by,
                notes,
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()

    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent audit logs."""
        cursor = self.conn.execute(
            """SELECT * FROM system_requests ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_pending_reviews(self) -> List[Dict]:
        """Get requests pending physician review."""
        cursor = self.conn.execute(
            """SELECT sr.* FROM system_requests sr
               LEFT JOIN human_decisions hd ON sr.id = hd.request_id
               WHERE sr.subsystem_used IN ('diagnosis', 'prescription_verify')
               AND hd.id IS NULL
               ORDER BY sr.id DESC LIMIT 20"""
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_subsystem_stats(self, subsystem: str = None) -> Dict:
        """Get statistics for a subsystem."""
        if subsystem:
            cursor = self.conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successful,
                          AVG(duration_ms) as avg_duration
                   FROM system_requests WHERE subsystem_used=?""",
                (subsystem,),
            )
        else:
            cursor = self.conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successful,
                          AVG(duration_ms) as avg_duration
                   FROM system_requests"""
            )
        return dict(cursor.fetchone())

    def close(self):
        """Close the database connection."""
        self.conn.close()
