import asyncio
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from agent.core import Agent
from database.schema import DatabaseManager  # Assuming it's imported

@dataclass
class AgentVisualState:
    position: tuple
    carrying: bool
    drives: Dict[str, float]
    justification: Dict[str, Any]

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, topic: str, callback: Callable):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def publish(self, topic: str, data: Any):
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                asyncio.create_task(callback(data))

event_bus = EventBus()

class AInceptionAdapter:
    def __init__(self, agent: Agent, db_path: str = "agent_state.db"):
        self.agent = agent
        self.db = DatabaseManager(db_path) if DatabaseManager else None
        self.current_state: Optional[AgentVisualState] = None

    async def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        result = self.agent.step(observation)
        self.update_state(result)
        event_bus.publish("agent_step", self.current_state)
        if self.db:
            self.log_to_db(result)
        return result

    def update_state(self, result: Dict[str, Any]):
        pos = result.get('observation', {}).get('agent_pos', (0, 0))
        carrying = result.get('observation', {}).get('carrying', False)
        drives = self.agent.drives.summary()
        justification = result.get('justification', {})
        self.current_state = AgentVisualState(pos, carrying, drives, justification)

    def log_to_db(self, result: Dict[str, Any]):
        # Placeholder for logging visual state to session_events
        pass

    def get_state(self) -> Optional[AgentVisualState]:
        return self.current_state
