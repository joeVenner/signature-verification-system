"""Tamper-Evident SHA-256 Hash-Chained Audit Logger.

Provides dual-persistence in SQLite and append-only JSONL formats.
Every clearing decision is cryptographically chained to its predecessor,
making post-hoc tampering or record modification mathematically detectable.
"""

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional, Tuple, List

from signature_verification_system.src.core.types import (
    AuditRecord,
    DecisionResult,
)
from signature_verification_system.src.core.config import SystemConfig, DEFAULT_CONFIG

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


def calculate_record_hash(
    prev_hash: str,
    sequence_num: int,
    timestamp_utc: str,
    document_id: str,
    amount: float,
    currency: str,
    similarity_score: float,
    decision_tier: str,
    action: str,
    policy_version: str,
    model_version: str,
    metadata_json: str
) -> str:
    """Calculate deterministic SHA-256 hash for an audit ledger block."""
    payload = (
        f"{prev_hash}|{sequence_num}|{timestamp_utc}|{document_id}|"
        f"{amount:.4f}|{currency}|{similarity_score:.4f}|{decision_tier}|"
        f"{action}|{policy_version}|{model_version}|{metadata_json}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLogger:
    """Tamper-evident audit ledger with SQLite and JSONL backends."""

    def __init__(
        self,
        db_path: str = "audit_ledger.db",
        jsonl_path: Optional[str] = "audit_ledger.jsonl",
        config: Optional[SystemConfig] = None
    ):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self.config = config or DEFAULT_CONFIG
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite audit schema with WAL mode."""
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_ledger (
                    sequence_num INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    timestamp_utc TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    decision_tier TEXT NOT NULL,
                    action TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    operator_id TEXT,
                    prev_hash TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_doc_id ON audit_ledger(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_seq ON audit_ledger(sequence_num);")
        conn.close()

    def get_last_record(self) -> Optional[AuditRecord]:
        """Fetch the most recent block in the hash chain."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT record_id, sequence_num, timestamp_utc, document_id, amount,
                   currency, similarity_score, decision_tier, action, policy_version,
                   model_version, operator_id, prev_hash, hash, metadata_json
            FROM audit_ledger
            ORDER BY sequence_num DESC
            LIMIT 1;
        """)
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return AuditRecord(
            record_id=row[0],
            sequence_num=row[1],
            timestamp_utc=row[2],
            document_id=row[3],
            amount=row[4],
            currency=row[5],
            similarity_score=row[6],
            decision_tier=row[7],
            action=row[8],
            policy_version=row[9],
            model_version=row[10],
            operator_id=row[11],
            prev_hash=row[12],
            hash=row[13],
            metadata=json.loads(row[14])
        )

    def get_recent_records(self, limit: int = 50, offset: int = 0) -> List[AuditRecord]:
        """Fetch recent blocks in the hash chain ordered by sequence descending."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT record_id, sequence_num, timestamp_utc, document_id, amount,
                   currency, similarity_score, decision_tier, action, policy_version,
                   model_version, operator_id, prev_hash, hash, metadata_json
            FROM audit_ledger
            ORDER BY sequence_num DESC
            LIMIT ? OFFSET ?;
        """, (limit, offset))
        rows = cursor.fetchall()
        conn.close()

        records: List[AuditRecord] = []
        for row in rows:
            meta = json.loads(row[14]) if row[14] else {}
            records.append(AuditRecord(
                record_id=row[0],
                sequence_num=row[1],
                timestamp_utc=row[2],
                document_id=row[3],
                amount=row[4],
                currency=row[5],
                similarity_score=row[6],
                decision_tier=row[7],
                action=row[8],
                policy_version=row[9],
                model_version=row[10],
                operator_id=row[11],
                prev_hash=row[12],
                hash=row[13],
                metadata=meta
            ))
        return records

    def get_record_count(self) -> int:
        """Return total count of records in the audit ledger."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM audit_ledger;")
        count = cursor.fetchone()[0]
        conn.close()
        return int(count)

    def log_decision(
        self,
        document_id: str,
        decision: DecisionResult,
        operator_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """Append an immutable clearing decision block to the ledger.
        
        Atomically queries the previous block hash and sequence number,
        computes the cryptographic SHA-256 block digest, and writes
        to both SQLite and append-only JSONL.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        record_id = str(uuid.uuid4())

        meta = {
            "reasons": decision.reasons,
            "cbuae_compliance_flags": decision.cbuae_compliance_flags,
            "return_code": decision.return_code.value if decision.return_code else None,
            "requires_four_eyes": decision.requires_four_eyes,
            **(additional_metadata or {})
        }
        metadata_json = json.dumps(meta, sort_keys=True)

        conn = sqlite3.connect(self.db_path)
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sequence_num, hash FROM audit_ledger ORDER BY sequence_num DESC LIMIT 1;")
            last_row = cursor.fetchone()

            if last_row:
                last_seq, prev_hash = last_row[0], last_row[1]
                new_seq = last_seq + 1
            else:
                new_seq = 1
                prev_hash = GENESIS_HASH

            block_hash = calculate_record_hash(
                prev_hash=prev_hash,
                sequence_num=new_seq,
                timestamp_utc=timestamp_utc,
                document_id=document_id,
                amount=decision.amount,
                currency=decision.currency,
                similarity_score=decision.similarity_score,
                decision_tier=decision.tier.value,
                action=decision.action,
                policy_version=decision.policy_version,
                model_version=self.config.model_version,
                metadata_json=metadata_json
            )

            cursor.execute("""
                INSERT INTO audit_ledger (
                    sequence_num, record_id, timestamp_utc, document_id, amount,
                    currency, similarity_score, decision_tier, action, policy_version,
                    model_version, operator_id, prev_hash, hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                new_seq, record_id, timestamp_utc, document_id, decision.amount,
                decision.currency, decision.similarity_score, decision.tier.value,
                decision.action, decision.policy_version, self.config.model_version,
                operator_id, prev_hash, block_hash, metadata_json
            ))
        conn.close()

        record = AuditRecord(
            record_id=record_id,
            sequence_num=new_seq,
            timestamp_utc=timestamp_utc,
            document_id=document_id,
            amount=decision.amount,
            currency=decision.currency,
            similarity_score=decision.similarity_score,
            decision_tier=decision.tier,
            action=decision.action,
            policy_version=decision.policy_version,
            model_version=self.config.model_version,
            operator_id=operator_id,
            prev_hash=prev_hash,
            hash=block_hash,
            metadata=meta
        )

        # Write to JSONL stream if specified
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.model_dump(), default=str) + "\n")

        return record

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify complete cryptographic chain integrity across all records.
        
        Returns:
            (True, None) if ledger is 100% untampered.
            (False, error_message) identifying the corrupted sequence number.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sequence_num, record_id, timestamp_utc, document_id, amount,
                   currency, similarity_score, decision_tier, action, policy_version,
                   model_version, operator_id, prev_hash, hash, metadata_json
            FROM audit_ledger
            ORDER BY sequence_num ASC;
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return True, None

        expected_prev_hash = GENESIS_HASH

        for row in rows:
            seq = row[0]
            ts = row[2]
            doc_id = row[3]
            amount = row[4]
            curr = row[5]
            score = row[6]
            tier = row[7]
            action = row[8]
            p_ver = row[9]
            m_ver = row[10]
            p_hash = row[12]
            curr_hash = row[13]
            meta_str = row[14]

            # 1. Verify link to previous block
            if p_hash != expected_prev_hash:
                return False, f"Broken chain at sequence {seq}: prev_hash '{p_hash}' != expected '{expected_prev_hash}'"

            # 2. Recompute current block hash
            recomputed = calculate_record_hash(
                prev_hash=p_hash,
                sequence_num=seq,
                timestamp_utc=ts,
                document_id=doc_id,
                amount=amount,
                currency=curr,
                similarity_score=score,
                decision_tier=tier,
                action=action,
                policy_version=p_ver,
                model_version=m_ver,
                metadata_json=meta_str
            )

            if recomputed != curr_hash:
                return False, f"Tampered record at sequence {seq}: stored hash '{curr_hash}' != computed '{recomputed}'"

            expected_prev_hash = curr_hash

        return True, None
