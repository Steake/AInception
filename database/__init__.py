"""
Database Package for AInception Agent

This package provides persistent storage for agent state including:
- Events and causal relationships
- Journal entries with natural language summaries  
- Action logs with justifications and drive states
- Promises and their lifecycle management
- Lessons learned from experience
- Reflex patches for dynamic behavior adaptation

Usage:
    from database import DatabaseManager, Event, ActionLog, Promise
    
    # Initialize database
    db = DatabaseManager("agent_state.db")
    
    # Create and save entities
    event = Event(event_type="move", delta={"position": (1,1)})
    event.save(db)
    
    # Log actions with justifications
    action_log = ActionLog(
        action={"type": "move"}, 
        justification={"reason": "approaching goal"}
    )
    action_log.save(db)
"""

from .schema import DatabaseManager, create_database
from .models import (
    Event,
    JournalEntry, 
    ActionLog,
    Promise,
    Lesson,
    create_models_with_db,
    get_agent_summary
)

__all__ = [
    'DatabaseManager',
    'create_database', 
    'Event',
    'JournalEntry',
    'ActionLog', 
    'Promise',
    'Lesson',
    'create_models_with_db',
    'get_agent_summary'
]
