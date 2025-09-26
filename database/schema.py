"""
Database Schema Definition for AInception Agent

This module defines the SQLite schema for persistent storage of:
- Events and their causal relationships
- Journal entries with natural language summaries
- Principles and their ranking changes over time
- Promises and their lifecycle management
- Action logs with justifications and drive states

Schema is designed to support:
1. Event-driven narrative journaling
2. Constitutional principle evolution tracking
3. Social promise enforcement and audit trails
4. Action decision justification and drive state monitoring
5. Surprise detection and lesson extraction
"""

import sqlite3
from typing import Optional, Dict, Any
import json
import time


# SQL Schema Definitions
SCHEMA_SQL = """
-- Events table: Core event stream with causal relationships
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    delta TEXT,  -- JSON serialized event data
    cause TEXT,
    consequence TEXT,
    surprise_level REAL DEFAULT 0.0,
    agent_tick INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Journal entries: Natural language narratives of events
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    event_id INTEGER,
    natural_text TEXT NOT NULL,
    summary_type TEXT DEFAULT 'immediate',  -- 'immediate', 'hourly', 'daily', 'lesson'
    agent_tick INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (event_id) REFERENCES events(id)
);

-- Visual sessions for AInceptionViz
CREATE TABLE IF NOT EXISTS visual_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    config_hash TEXT,
    episode_count INTEGER,
    ml_models_used TEXT,  -- JSON
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Session events for visualization data
CREATE TABLE IF NOT EXISTS session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    tick INTEGER,
    agent_state TEXT,  -- JSON
    ml_predictions TEXT,  -- JSON
    visual_frame TEXT,  -- Path to frame
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES visual_sessions(id)
);

-- Principles: Constitutional principles with ranking
CREATE TABLE IF NOT EXISTS principles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    principle TEXT NOT NULL UNIQUE,
    rank INTEGER NOT NULL,
    description TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    modified_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Principle changes: Track constitutional re-rankings with proofs
CREATE TABLE IF NOT EXISTS principle_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    old_ranking TEXT,  -- JSON serialized list of principles with ranks
    new_ranking TEXT,  -- JSON serialized list of principles with ranks
    proof TEXT,        -- JSON serialized proof object
    agent_tick INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Promises: Social promise registration and lifecycle
CREATE TABLE IF NOT EXISTS promises (
    id TEXT PRIMARY KEY,
    condition TEXT NOT NULL,
    expected_behavior TEXT NOT NULL,
    expiry INTEGER NOT NULL,
    penalty_function TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'fulfilled', 'breached', 'expired'
    created_at INTEGER NOT NULL,
    resolved_at INTEGER,
    breach_reason TEXT,
    fulfill_reason TEXT,
    override_proof TEXT,  -- JSON serialized proof for overrides
    agent_tick_created INTEGER,
    agent_tick_resolved INTEGER
);

-- Action logs: Decision justifications and drive states
CREATE TABLE IF NOT EXISTS action_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    agent_tick INTEGER NOT NULL,
    action TEXT NOT NULL,           -- JSON serialized action
    justification TEXT NOT NULL,    -- JSON serialized justification
    drive_errors TEXT NOT NULL,     -- JSON serialized drive error values
    principles_checked TEXT,        -- JSON serialized principles referenced
    promise_violations TEXT,        -- JSON serialized promise violation info
    alternative_actions TEXT,       -- JSON serialized alternative candidates considered
    cost_breakdown TEXT,            -- JSON serialized cost component breakdown
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Lessons: Extracted lessons from surprise events
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    agent_tick INTEGER NOT NULL,
    lesson_type TEXT NOT NULL,      -- 'surprise', 'pattern', 'principle_conflict', etc.
    lesson_text TEXT NOT NULL,
    related_events TEXT,            -- JSON serialized list of event IDs
    confidence REAL DEFAULT 0.5,
    applied_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Reflex patches: Dynamic reflex rule additions
CREATE TABLE IF NOT EXISTS reflex_patches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    agent_tick INTEGER NOT NULL,
    state_signature TEXT NOT NULL,  -- Key for when this patch applies
    rule_description TEXT NOT NULL,
    condition_code TEXT,            -- Serialized condition function
    action_code TEXT,               -- Serialized action function
    triggered_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_agent_tick ON events(agent_tick);
CREATE INDEX IF NOT EXISTS idx_journal_timestamp ON journal(timestamp);
CREATE INDEX IF NOT EXISTS idx_journal_event_id ON journal(event_id);
CREATE INDEX IF NOT EXISTS idx_action_logs_timestamp ON action_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_action_logs_agent_tick ON action_logs(agent_tick);
CREATE INDEX IF NOT EXISTS idx_promises_status ON promises(status);
CREATE INDEX IF NOT EXISTS idx_lessons_timestamp ON lessons(timestamp);
CREATE INDEX IF NOT EXISTS idx_lessons_type ON lessons(lesson_type);
"""


class DatabaseManager:
    """
    Database manager for AInception agent persistence.
    Provides high-level interface for storing and retrieving agent state.
    """
    
    def __init__(self, db_path: str = "agent_state.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._initialize_db()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating if necessary."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row  # Enable column access by name
        return self._connection
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def _initialize_db(self):
        """Initialize database schema."""
        conn = self.get_connection()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    
    # -------------------- Event Management --------------------
    
    def insert_event(
        self,
        event_type: str,
        delta: Dict[str, Any],
        cause: str = "",
        consequence: str = "",
        surprise_level: float = 0.0,
        agent_tick: int = 0,
        timestamp: Optional[int] = None
    ) -> int:
        """Insert new event and return event ID."""
        if timestamp is None:
            timestamp = int(time.time())
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO events (timestamp, event_type, delta, cause, consequence, surprise_level, agent_tick)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, event_type, json.dumps(delta), cause, consequence, surprise_level, agent_tick))
        
        event_id = cursor.lastrowid
        conn.commit()
        return event_id
    
    def get_events(
        self,
        since_timestamp: Optional[int] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list:
        """Retrieve events with optional filtering."""
        conn = self.get_connection()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if since_timestamp:
            query += " AND timestamp >= ?"
            params.append(since_timestamp)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    # -------------------- Journal Management --------------------
    
    def insert_journal_entry(
        self,
        natural_text: str,
        event_id: Optional[int] = None,
        summary_type: str = "immediate",
        agent_tick: int = 0,
        timestamp: Optional[int] = None
    ) -> int:
        """Insert journal entry and return entry ID."""
        if timestamp is None:
            timestamp = int(time.time())
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO journal (timestamp, event_id, natural_text, summary_type, agent_tick)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, event_id, natural_text, summary_type, agent_tick))
        
        entry_id = cursor.lastrowid
        conn.commit()
        return entry_id
    
    # -------------------- Action Logging --------------------
    
    def log_action(
        self,
        agent_tick: int,
        action: Dict[str, Any],
        justification: Dict[str, Any],
        drive_errors: Dict[str, float],
        principles_checked: Optional[list] = None,
        promise_violations: Optional[list] = None,
        alternative_actions: Optional[list] = None,
        cost_breakdown: Optional[Dict[str, float]] = None,
        timestamp: Optional[int] = None
    ) -> int:
        """Log action decision with full context."""
        if timestamp is None:
            timestamp = int(time.time())
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO action_logs (
                timestamp, agent_tick, action, justification, drive_errors,
                principles_checked, promise_violations, alternative_actions, cost_breakdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, agent_tick,
            json.dumps(action),
            json.dumps(justification),
            json.dumps(drive_errors),
            json.dumps(principles_checked or []),
            json.dumps(promise_violations or []),
            json.dumps(alternative_actions or []),
            json.dumps(cost_breakdown or {})
        ))
        
        action_id = cursor.lastrowid
        conn.commit()
        return action_id
    
    # -------------------- Promise Management --------------------
    
    def insert_promise(
        self,
        promise_id: str,
        condition: str,
        expected_behavior: str,
        expiry: int,
        penalty_function: str,
        agent_tick: int = 0,
        timestamp: Optional[int] = None
    ):
        """Insert new promise."""
        if timestamp is None:
            timestamp = int(time.time())
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO promises (
                id, condition, expected_behavior, expiry, penalty_function,
                status, created_at, agent_tick_created
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
        """, (promise_id, condition, expected_behavior, expiry, penalty_function, timestamp, agent_tick))
        
        conn.commit()
    
    def update_promise_status(
        self,
        promise_id: str,
        status: str,
        agent_tick: int = 0,
        reason: str = "",
        timestamp: Optional[int] = None
    ):
        """Update promise status (fulfilled, breached, expired)."""
        if timestamp is None:
            timestamp = int(time.time())
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Update based on status
        if status == "fulfilled":
            cursor.execute("""
                UPDATE promises SET status = ?, resolved_at = ?, 
                       fulfill_reason = ?, agent_tick_resolved = ?
                WHERE id = ?
            """, (status, timestamp, reason, agent_tick, promise_id))
        elif status == "breached":
            cursor.execute("""
                UPDATE promises SET status = ?, resolved_at = ?, 
                       breach_reason = ?, agent_tick_resolved = ?
                WHERE id = ?
            """, (status, timestamp, reason, agent_tick, promise_id))
        else:
            cursor.execute("""
                UPDATE promises SET status = ?, resolved_at = ?, agent_tick_resolved = ?
                WHERE id = ?
            """, (status, timestamp, agent_tick, promise_id))
        
        conn.commit()
    
    # -------------------- Utility Methods --------------------
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        tables = ["events", "journal", "action_logs", "promises", "lessons", "reflex_patches"]
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove old data beyond specified retention period."""
        cutoff_timestamp = int(time.time()) - (days_to_keep * 24 * 60 * 60)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Clean up old events and journal entries
        cursor.execute("DELETE FROM journal WHERE timestamp < ?", (cutoff_timestamp,))
        cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_timestamp,))
        cursor.execute("DELETE FROM action_logs WHERE timestamp < ?", (cutoff_timestamp,))
        
        conn.commit()


# Convenience functions for backward compatibility
def create_database(db_path: str = "agent_state.db") -> DatabaseManager:
    """Create and initialize database."""
    return DatabaseManager(db_path)


if __name__ == "__main__":
    # Test database creation and basic operations
    db = DatabaseManager(":memory:")  # In-memory database for testing
    
    # Test event insertion
    event_id = db.insert_event(
        event_type="test_event",
        delta={"test": "data"},
        cause="test",
        agent_tick=1
    )
    print(f"Created event {event_id}")
    
    # Test journal entry
    journal_id = db.insert_journal_entry(
        natural_text="This is a test journal entry",
        event_id=event_id,
        agent_tick=1
    )
    print(f"Created journal entry {journal_id}")
    
    # Test action logging
    action_id = db.log_action(
        agent_tick=1,
        action={"type": "move", "position": (1, 0)},
        justification={"reason": "testing"},
        drive_errors={"energy": -0.1}
    )
    print(f"Logged action {action_id}")
    
    # Test promise insertion
    db.insert_promise(
        promise_id="P001",
        condition="avoid:(3,3)",
        expected_behavior="Do not enter tile (3,3)",
        expiry=100,
        penalty_function="cost:5.0",
        agent_tick=1
    )
    print("Created promise P001")
    
    # Get stats
    stats = db.get_stats()
    print(f"Database stats: {stats}")
    
    db.close()
    print("Database test completed successfully!")
