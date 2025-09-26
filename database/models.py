"""
Database Models and Access Layer

This module provides high-level model classes for interacting with the database.
It abstracts the SQL layer and provides object-oriented access to agent data.

Models included:
- Event: Core event with causal relationships
- JournalEntry: Natural language narrative entry
- ActionLog: Decision justification with drive state
- Promise: Social promise with lifecycle management
- Lesson: Extracted learning from experience
- ReflexPatch: Dynamic reflex rule addition

Each model provides:
- Object-oriented access to database records
- Validation and serialization
- Relationships between entities
- Query methods for common operations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import time

# Handle imports for both package and standalone usage
try:
    from .schema import DatabaseManager
except ImportError:
    from schema import DatabaseManager


@dataclass
class Event:
    """
    Represents a single event in the agent's experience.
    Events are the core building blocks of the narrative journal.
    """
    id: Optional[int] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    event_type: str = ""
    delta: Dict[str, Any] = field(default_factory=dict)
    cause: str = ""
    consequence: str = ""
    surprise_level: float = 0.0
    agent_tick: int = 0
    created_at: Optional[int] = None
    
    def save(self, db: DatabaseManager) -> int:
        """Save event to database and return event ID."""
        if self.id is None:
            self.id = db.insert_event(
                event_type=self.event_type,
                delta=self.delta,
                cause=self.cause,
                consequence=self.consequence,
                surprise_level=self.surprise_level,
                agent_tick=self.agent_tick,
                timestamp=self.timestamp
            )
        return self.id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Create Event from database row."""
        delta = json.loads(data.get('delta', '{}')) if isinstance(data.get('delta'), str) else data.get('delta', {})
        return cls(
            id=data.get('id'),
            timestamp=data.get('timestamp', int(time.time())),
            event_type=data.get('event_type', ''),
            delta=delta,
            cause=data.get('cause', ''),
            consequence=data.get('consequence', ''),
            surprise_level=data.get('surprise_level', 0.0),
            agent_tick=data.get('agent_tick', 0),
            created_at=data.get('created_at')
        )
    
    @classmethod
    def get_recent(cls, db: DatabaseManager, limit: int = 10) -> List[Event]:
        """Get recent events from database."""
        rows = db.get_events(limit=limit)
        return [cls.from_dict(row) for row in rows]
    
    @classmethod
    def get_by_type(cls, db: DatabaseManager, event_type: str, limit: int = 50) -> List[Event]:
        """Get events of specific type."""
        rows = db.get_events(event_type=event_type, limit=limit)
        return [cls.from_dict(row) for row in rows]


@dataclass
class JournalEntry:
    """
    Represents a journal entry with natural language narrative.
    Journal entries provide human-readable summaries of events and experiences.
    """
    id: Optional[int] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    event_id: Optional[int] = None
    natural_text: str = ""
    summary_type: str = "immediate"  # 'immediate', 'hourly', 'daily', 'lesson'
    agent_tick: int = 0
    created_at: Optional[int] = None
    
    def save(self, db: DatabaseManager) -> int:
        """Save journal entry to database."""
        if self.id is None:
            self.id = db.insert_journal_entry(
                natural_text=self.natural_text,
                event_id=self.event_id,
                summary_type=self.summary_type,
                agent_tick=self.agent_tick,
                timestamp=self.timestamp
            )
        return self.id


@dataclass
class ActionLog:
    """
    Represents a logged action decision with full justification context.
    Enables auditing and analysis of agent decision-making.
    """
    id: Optional[int] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    agent_tick: int = 0
    action: Dict[str, Any] = field(default_factory=dict)
    justification: Dict[str, Any] = field(default_factory=dict)
    drive_errors: Dict[str, float] = field(default_factory=dict)
    principles_checked: List[str] = field(default_factory=list)
    promise_violations: List[str] = field(default_factory=list)
    alternative_actions: List[Dict[str, Any]] = field(default_factory=list)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    created_at: Optional[int] = None
    
    def save(self, db: DatabaseManager) -> int:
        """Save action log to database."""
        if self.id is None:
            self.id = db.log_action(
                agent_tick=self.agent_tick,
                action=self.action,
                justification=self.justification,
                drive_errors=self.drive_errors,
                principles_checked=self.principles_checked,
                promise_violations=self.promise_violations,
                alternative_actions=self.alternative_actions,
                cost_breakdown=self.cost_breakdown,
                timestamp=self.timestamp
            )
        return self.id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionLog:
        """Create ActionLog from database row."""
        return cls(
            id=data.get('id'),
            timestamp=data.get('timestamp', int(time.time())),
            agent_tick=data.get('agent_tick', 0),
            action=json.loads(data.get('action', '{}')) if isinstance(data.get('action'), str) else data.get('action', {}),
            justification=json.loads(data.get('justification', '{}')) if isinstance(data.get('justification'), str) else data.get('justification', {}),
            drive_errors=json.loads(data.get('drive_errors', '{}')) if isinstance(data.get('drive_errors'), str) else data.get('drive_errors', {}),
            principles_checked=json.loads(data.get('principles_checked', '[]')) if isinstance(data.get('principles_checked'), str) else data.get('principles_checked', []),
            promise_violations=json.loads(data.get('promise_violations', '[]')) if isinstance(data.get('promise_violations'), str) else data.get('promise_violations', []),
            alternative_actions=json.loads(data.get('alternative_actions', '[]')) if isinstance(data.get('alternative_actions'), str) else data.get('alternative_actions', []),
            cost_breakdown=json.loads(data.get('cost_breakdown', '{}')) if isinstance(data.get('cost_breakdown'), str) else data.get('cost_breakdown', {}),
            created_at=data.get('created_at')
        )


@dataclass 
class Promise:
    """
    Represents a social promise with lifecycle management.
    Promises are commitments made by the agent that constrain future behavior.
    """
    id: str = ""
    condition: str = ""
    expected_behavior: str = ""
    expiry: int = 0
    penalty_function: str = ""
    status: str = "active"  # 'active', 'fulfilled', 'breached', 'expired'
    created_at: int = field(default_factory=lambda: int(time.time()))
    resolved_at: Optional[int] = None
    breach_reason: str = ""
    fulfill_reason: str = ""
    override_proof: Optional[Dict[str, Any]] = None
    agent_tick_created: int = 0
    agent_tick_resolved: Optional[int] = None
    
    def save(self, db: DatabaseManager):
        """Save promise to database."""
        db.insert_promise(
            promise_id=self.id,
            condition=self.condition,
            expected_behavior=self.expected_behavior,
            expiry=self.expiry,
            penalty_function=self.penalty_function,
            agent_tick=self.agent_tick_created,
            timestamp=self.created_at
        )
    
    def fulfill(self, db: DatabaseManager, reason: str = "", agent_tick: int = 0):
        """Mark promise as fulfilled."""
        self.status = "fulfilled"
        self.fulfill_reason = reason
        self.resolved_at = int(time.time())
        self.agent_tick_resolved = agent_tick
        
        db.update_promise_status(
            promise_id=self.id,
            status=self.status,
            agent_tick=agent_tick,
            reason=reason
        )
    
    def breach(self, db: DatabaseManager, reason: str = "", agent_tick: int = 0):
        """Mark promise as breached."""
        self.status = "breached"
        self.breach_reason = reason
        self.resolved_at = int(time.time())
        self.agent_tick_resolved = agent_tick
        
        db.update_promise_status(
            promise_id=self.id,
            status=self.status,
            agent_tick=agent_tick,
            reason=reason
        )
    
    def expire(self, db: DatabaseManager, agent_tick: int = 0):
        """Mark promise as expired."""
        self.status = "expired"
        self.resolved_at = int(time.time())
        self.agent_tick_resolved = agent_tick
        
        db.update_promise_status(
            promise_id=self.id,
            status=self.status,
            agent_tick=agent_tick
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Promise:
        """Create Promise from database row."""
        override_proof = None
        if data.get('override_proof'):
            override_proof = json.loads(data['override_proof']) if isinstance(data['override_proof'], str) else data['override_proof']
        
        return cls(
            id=data.get('id', ''),
            condition=data.get('condition', ''),
            expected_behavior=data.get('expected_behavior', ''),
            expiry=data.get('expiry', 0),
            penalty_function=data.get('penalty_function', ''),
            status=data.get('status', 'active'),
            created_at=data.get('created_at', int(time.time())),
            resolved_at=data.get('resolved_at'),
            breach_reason=data.get('breach_reason', ''),
            fulfill_reason=data.get('fulfill_reason', ''),
            override_proof=override_proof,
            agent_tick_created=data.get('agent_tick_created', 0),
            agent_tick_resolved=data.get('agent_tick_resolved')
        )


@dataclass
class Lesson:
    """
    Represents a learned lesson from agent experience.
    Lessons capture insights that can influence future behavior.
    """
    id: Optional[int] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    agent_tick: int = 0
    lesson_type: str = ""  # 'surprise', 'pattern', 'principle_conflict', etc.
    lesson_text: str = ""
    related_events: List[int] = field(default_factory=list)
    confidence: float = 0.5
    applied_count: int = 0
    success_count: int = 0
    created_at: Optional[int] = None
    
    def save(self, db: DatabaseManager) -> int:
        """Save lesson to database."""
        if self.id is None:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO lessons (
                    timestamp, agent_tick, lesson_type, lesson_text, 
                    related_events, confidence, applied_count, success_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.timestamp, self.agent_tick, self.lesson_type, self.lesson_text,
                json.dumps(self.related_events), self.confidence, self.applied_count, self.success_count
            ))
            self.id = cursor.lastrowid
            conn.commit()
        return self.id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Lesson:
        """Create Lesson from database row."""
        related_events = json.loads(data.get('related_events', '[]')) if isinstance(data.get('related_events'), str) else data.get('related_events', [])
        
        return cls(
            id=data.get('id'),
            timestamp=data.get('timestamp', int(time.time())),
            agent_tick=data.get('agent_tick', 0),
            lesson_type=data.get('lesson_type', ''),
            lesson_text=data.get('lesson_text', ''),
            related_events=related_events,
            confidence=data.get('confidence', 0.5),
            applied_count=data.get('applied_count', 0),
            success_count=data.get('success_count', 0),
            created_at=data.get('created_at')
        )


# Utility functions for common database operations
def create_models_with_db(db_path: str = "agent_state.db") -> DatabaseManager:
    """Create database manager and initialize schema."""
    return DatabaseManager(db_path)


def get_agent_summary(db: DatabaseManager, agent_tick: int) -> Dict[str, Any]:
    """Get comprehensive summary of agent state at specific tick."""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Get recent events
    cursor.execute("""
        SELECT * FROM events WHERE agent_tick <= ? 
        ORDER BY agent_tick DESC LIMIT 10
    """, (agent_tick,))
    recent_events = [Event.from_dict(dict(row)) for row in cursor.fetchall()]
    
    # Get recent actions
    cursor.execute("""
        SELECT * FROM action_logs WHERE agent_tick <= ?
        ORDER BY agent_tick DESC LIMIT 5
    """, (agent_tick,))
    recent_actions = [ActionLog.from_dict(dict(row)) for row in cursor.fetchall()]
    
    # Get active promises
    cursor.execute("""
        SELECT * FROM promises WHERE status = 'active'
        AND agent_tick_created <= ?
    """, (agent_tick,))
    active_promises = [Promise.from_dict(dict(row)) for row in cursor.fetchall()]
    
    # Get recent lessons
    cursor.execute("""
        SELECT * FROM lessons WHERE agent_tick <= ?
        ORDER BY agent_tick DESC LIMIT 5
    """, (agent_tick,))
    recent_lessons = [Lesson.from_dict(dict(row)) for row in cursor.fetchall()]
    
    return {
        "agent_tick": agent_tick,
        "recent_events": [event.__dict__ for event in recent_events],
        "recent_actions": [action.__dict__ for action in recent_actions],
        "active_promises": [promise.__dict__ for promise in active_promises],
        "recent_lessons": [lesson.__dict__ for lesson in recent_lessons],
        "stats": db.get_stats()
    }


if __name__ == "__main__":
    # Test model operations
    db = DatabaseManager(":memory:")
    
    # Test event creation
    event = Event(
        event_type="test_move",
        delta={"position": (1, 1), "energy": -0.01},
        cause="user_command",
        agent_tick=1
    )
    event_id = event.save(db)
    print(f"Created event {event_id}")
    
    # Test action log
    action_log = ActionLog(
        agent_tick=1,
        action={"type": "move", "position": (1, 1)},
        justification={"reason": "moving toward goal"},
        drive_errors={"energy": -0.1},
        principles_checked=["efficiency", "safety"]
    )
    action_id = action_log.save(db)
    print(f"Created action log {action_id}")
    
    # Test promise
    promise = Promise(
        id="P001",
        condition="avoid:(3,3)",
        expected_behavior="Do not enter forbidden tile",
        expiry=100,
        penalty_function="cost:5.0",
        agent_tick_created=1
    )
    promise.save(db)
    promise.fulfill(db, "Successfully avoided area", agent_tick=50)
    print(f"Created and fulfilled promise {promise.id}")
    
    # Test lesson
    lesson = Lesson(
        agent_tick=1,
        lesson_type="surprise",
        lesson_text="Movement cost was higher than expected in this area",
        related_events=[event_id],
        confidence=0.8
    )
    lesson_id = lesson.save(db)
    print(f"Created lesson {lesson_id}")
    
    # Get summary
    summary = get_agent_summary(db, agent_tick=1)
    print(f"Agent summary: {len(summary['recent_events'])} events, {len(summary['recent_actions'])} actions")
    
    db.close()
    print("Model test completed successfully!")
