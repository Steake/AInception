"""
Core Agent Orchestration

Responsibilities:
- Initialize subsystems (drives, perception/eventifier, constitution, social promises, reflex, planner, imagination stub)
- Provide step(observation) API returning {action, justification, logs}
- Maintain global tick counter and lightweight internal state
- Surface helper APIs: register_promise, get_drive_summary, get_principles, set_goal
- Hook surprise / lesson pathway (placeholder for later integration)

Assumptions / Simplifications (MVP):
- Observation dict may include:
  {
    "agent_pos": (x,y),
    "goal": (xg, yg),              # optional
    "energy": float,
    "temperature": float,
    "social_proximity": float,
    "danger_tiles": set((x,y),...),
    "forbidden_tiles": set((x,y),...)
  }
- Planner operates over grid positions contained in observation.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple, Callable
import os
import yaml
import time

from .drives import build_from_config as build_drive_system, DriveSystem
from .perception import Eventifier
from .journal import Journal, JournalConfig
from .constitution import Constitution
from .social import PromiseBook
from .imagination import Imagination
from .policy.reflex import ReflexLayer
from .policy.planner import (
    Planner,
    PlannerConfig,
    CostWeights,
    default_get_neighbors,
    always_safe,
    dummy_drive_errors,
    simple_goal_fn,
    extract_pos,
    NeuralModuleInterface
)

# Import database components
try:
    from database import DatabaseManager, ActionLog, Event
except ImportError:
    # Fallback for when database isn't available
    DatabaseManager = None
    ActionLog = None 
    Event = None


# ---------------------------------------------------------------------------
# Utility: Config Loading
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Neural Module Stub (Optional Injection)
# ---------------------------------------------------------------------------

class NeuralStub(NeuralModuleInterface):
    """
    Placeholder neural augmentation module.
    Returns zero-cost, low confidence predictions until replaced.
    """
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def predict_state(self, state: Dict[str, Any]):
        if not self.enabled:
            return super().predict_state(state)
        # Minimal heuristic: if energy low, increase predicted drive cost.
        energy = state.get("energy", 0.7)
        drive_cost = max(0.0, 0.7 - energy)
        return type(super().predict_state(state))(
            drive_cost=drive_cost,
            risk_score=0.0,
            energy_delta=0.0,
            confidence=0.6 if self.enabled else 0.0
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        config_dir: str = "config",
        enable_journal_llm: bool = False,
        enable_neural: bool = False,
        db_path: str = "agent_state.db"
    ):
        self.config_dir = config_dir
        self.tick: int = 0
        self._goal: Optional[Tuple[int, int]] = None

        # Initialize database (optional)
        self.db = None
        if DatabaseManager:
            try:
                self.db = DatabaseManager(db_path)
            except Exception as e:
                print(f"Warning: Could not initialize database: {e}")

        # Load configurations
        drives_cfg_path = os.path.join(config_dir, "drives.yaml")
        principles_cfg_path = os.path.join(config_dir, "principles.yaml")

        self.drive_cfg = _load_yaml(drives_cfg_path)
        self.principles_cfg = _load_yaml(principles_cfg_path)

        # Subsystems
        self.drives: DriveSystem = build_drive_system(self.drive_cfg)
        self.eventifier = Eventifier(window=8)
        
        # Create journal config and pass to journal
        journal_config = None
        if not enable_journal_llm:
            journal_config = JournalConfig(use_llm=False)
        self.journal = Journal(config=journal_config, db_conn=self.db.get_connection() if self.db else None)
        
        self.constitution = Constitution(self.principles_cfg.get("principles", []))
        self.social = PromiseBook()
        self.imagination = Imagination(horizons=(1, 3))
        self.reflex = ReflexLayer()

        # Neural stub (replace later with actual implementation)
        self.neural = NeuralStub(enabled=enable_neural) if enable_neural else None

        # Planner configuration & instantiation
        planner_cfg = PlannerConfig(
            candidate_k=6,
            prefix_h=6,
            imagination_horizons=(1, 3),
            use_neural=enable_neural
        )

        def _drive_error_fn(state: Dict[str, Any]):
            # Use live drives to produce error vector
            return self.drives.drive_errors()

        def _goal_fn(state: Dict[str, Any]):
            # Goal can be set explicitly or come from observation
            return self._goal or state.get("goal")

        def _is_principle_violation(node):
            # Placeholder: user can add real checks (danger tiles, etc.)
            forbidden = self._last_observation.get("forbidden_tiles") if self._last_observation else None
            if forbidden and node in forbidden:
                return True
            return False

        def _is_promise_violation(node):
            # Use PromiseBook's safe node violation checking
            return self.social.node_violates(node)

        self.planner = Planner(
            config=planner_cfg,
            cost_weights=CostWeights(),
            get_neighbors=default_get_neighbors,
            is_principle_violation=_is_principle_violation,
            is_promise_violation=_is_promise_violation,
            drive_error_fn=_drive_error_fn,
            goal_fn=_goal_fn,
            state_position_extractor=lambda s: s.get("agent_pos", (0, 0)),
            neural_module=self.neural,
            promise_book=self.social,
            imagination=self.imagination,
            drive_system=self.drives
        )

        # Internal caches
        self._prediction_cache: Dict[str, Any] = {}
        self._last_observation: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_goal(self, goal: Tuple[int, int]):
        self._goal = goal

    def register_promise(self, condition: str, behavior: str, expiry: int, penalty: str) -> str:
        pid = self.social.register(condition, behavior, expiry, penalty)
        self._journal_event(
            {
                "type": "promise_registered",
                "promise_id": pid,
                "condition": condition,
                "behavior": behavior
            }
        )
        return pid

    def get_drive_summary(self) -> Dict[str, Dict]:
        return self.drives.summary()

    def get_principles(self) -> List[Tuple[int, str]]:
        return self.constitution.ranking()

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main control tick.
        1. Update internal tick and ingest observation
        2. Update drives (decay + ingestion)
        3. Perception window update
        4. Reflex attempt
        5. If no reflex, deliberate planning
        6. Log justification + event extraction
        """
        self.tick += 1
        self._last_observation = observation

        # Drives
        self.drives.tick_decay()
        self.drives.ingest_observation(observation)

        # Perception / feature extraction
        features = self.eventifier.observe(observation)
        events = self.eventifier.extract_events()
        for e in events:
            self._journal_event(e)

        # Reflex attempt
        drive_errors_current = self.drives.drive_errors()
        reflex_action = self.reflex.maybe_act(features, drive_errors_current)
        if reflex_action is not None:
            justification = {
                "chosen_goal": None,
                "principles_checked": [p[1] for p in self.constitution.ranking()[:3]],
                "tradeoffs": {},
                "type": "reflex",
                "dominant_drive": max(drive_errors_current.items(), key=lambda kv: abs(kv[1]))[0] if drive_errors_current else None
            }
            logs = self._compose_logs(reflex_action, justification, reflex=True)
            # Journal the reflex decision explicitly
            self.journal.log_action(self.tick, reflex_action, justification, drive_errors_current)
            return {
                "action": reflex_action,
                "justification": justification,
                "logs": logs
            }

        # Planner state assembly
        planner_state = {
            "agent_pos": observation.get("agent_pos", (0, 0)),
            "goal": self._goal or observation.get("goal"),
            "energy": observation.get("energy", self.drives.drives.get("energy", None).current if "energy" in self.drives.drives else 0.0),
            "temperature": observation.get("temperature", self.drives.drives.get("temperature", None).current if "temperature" in self.drives.drives else 0.0),
            "social_proximity": observation.get("social_proximity", self.drives.drives.get("social_proximity", None).current if "social_proximity" in self.drives.drives else 0.0),
        }

        plan_out = self.planner.plan(planner_state)
        action = plan_out["action"]
        justification = plan_out["justification"]
        # Journal the deliberative action decision
        self.journal.log_action(self.tick, action, justification, self.drives.drive_errors())

        # Surprise handling placeholder (real comparison after environment step)
        justification["surprise_flag"] = False

        logs = self._compose_logs(action, justification)
        return {
            "action": action,
            "justification": justification,
            "logs": logs
        }

    # ---------------------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------------------
    def _journal_event(self, event: Dict[str, Any]):
        enriched = {
            "timestamp": self.tick,
            **event
        }
        self.journal.append_event(enriched)

    def _compose_logs(self, action: Dict[str, Any], justification: Dict[str, Any], reflex: bool = False) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "action": action,
            "reflex": reflex,
            "drives": self.drives.summary(),
            "principles_top3": [p[1] for p in self.constitution.ranking()[:3]],
            "promise_count": len(self.social.promises),
            "justification": justification
        }

    # ---------------------------------------------------------------------
    # Serialization (partial)
    # ---------------------------------------------------------------------
    def save_state(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "drives": self.drives.to_dict(),
            "goal": self._goal
        }

    def load_state(self, state: Dict[str, Any]):
        self.tick = state.get("tick", self.tick)
        if "drives" in state:
            self.drives.from_dict(state["drives"])
        self._goal = state.get("goal", self._goal)


# ---------------------------------------------------------------------------
# Convenience Factory
# ---------------------------------------------------------------------------

def build_agent(config_dir: str = "config", enable_journal_llm: bool = False, enable_neural: bool = False) -> Agent:
    return Agent(
        config_dir=config_dir,
        enable_journal_llm=enable_journal_llm,
        enable_neural=enable_neural
    )


# ---------------------------------------------------------------------------
# Basic Self-Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = build_agent()
    agent.set_goal((3, 0))
    obs = {
        "agent_pos": (0, 0),
        "energy": 0.65,
        "temperature": 0.5,
        "social_proximity": 0.2
    }
    out = agent.step(obs)
    print("STEP OUTPUT:", out)