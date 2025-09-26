"""
Perception & Event Extraction / Causal Attribution

Responsibilities:
- Maintain rolling window of recent observations + actions
- Produce feature dict for policy stack
- Extract discrete events:
    * state_change (drive threshold crossings, goal proximity changes)
    * resource_gain / resource_loss (energy deltas)
    * help / harm (temperature extremes, social proximity improvements/drops)
    * goal_contact (agent enters goal tile)
- Simple causal attribution:
    * Link latest action to resulting deltas if temporal adjacency (1 tick lag) and pattern matches expected effect rules
- Provide compact state signature for reflex patching & surprise tracking

Observation Conventions (partial, flexible):
{
  "agent_pos": (x, y),
  "goal": (gx, gy) | None,
  "energy": float,
  "temperature": float,
  "social_proximity": float,
  "danger_tiles": set((x,y), ...),
  "forbidden_tiles": set((x,y), ...),
  "tick": int (optional; will be injected externally if absent)
}

Action (optional for perception.feed_action):
{
  "type": "move" | "noop" | ...,
  "dx": int,
  "dy": int
}

Design Notes:
- Keep purely in-memory; persistence handled by journal & DB layers.
- Avoid heavy dependencies; numpy optional but not required.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Deque
from collections import deque
import math
import hashlib


class Eventifier:
    def __init__(
        self,
        window: int = 12,
        drive_keys: Optional[List[str]] = None,
        change_eps: float = 1e-3,
        energy_gain_threshold: float = 0.02,
        energy_loss_threshold: float = -0.02
    ):
        self.window = window
        self.obs_history: Deque[Dict[str, Any]] = deque(maxlen=window)
        self.action_history: Deque[Dict[str, Any]] = deque(maxlen=window)
        self.events_buffer: List[Dict[str, Any]] = []
        self.drive_keys = drive_keys or ["energy", "temperature", "social_proximity"]
        self.change_eps = change_eps
        self.energy_gain_threshold = energy_gain_threshold
        self.energy_loss_threshold = energy_loss_threshold

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------
    def observe(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest observation; return feature summary for policy stack."""
        obs = dict(obs)  # shallow copy
        if "tick" not in obs:
            obs["tick"] = len(self.obs_history)
        self._append_observation(obs)
        self._derive_events()
        features = self._build_features(obs)
        return features

    def feed_action(self, action: Dict[str, Any]):
        """Register last executed action for causal linkage."""
        self.action_history.append(action)

    def extract_events(self) -> List[Dict[str, Any]]:
        """Drain and return accumulated events since last call."""
        out = self.events_buffer
        self.events_buffer = []
        return out

    def recent(self, n: int = 5) -> List[Dict[str, Any]]:
        return list(self.obs_history)[-n:]

    # ------------------------------------------------------------------
    # Internal: Observation Bookkeeping
    # ------------------------------------------------------------------
    def _append_observation(self, obs: Dict[str, Any]):
        self.obs_history.append(obs)

    # ------------------------------------------------------------------
    # Feature Construction
    # ------------------------------------------------------------------
    def _build_features(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        prev = self._prev_obs()
        energy_delta = 0.0
        if prev is not None and "energy" in obs and "energy" in prev:
            energy_delta = obs["energy"] - prev["energy"]

        distance_to_goal = None
        if obs.get("goal") and obs.get("agent_pos"):
            distance_to_goal = self._manhattan(obs["agent_pos"], obs["goal"])

        # Drive snapshot
        drives = {
            k: obs.get(k, None)
            for k in self.drive_keys
            if k in obs
        }

        features = {
            "tick": obs["tick"],
            "agent_pos": obs.get("agent_pos"),
            "goal": obs.get("goal"),
            "distance_to_goal": distance_to_goal,
            "energy_delta": energy_delta,
            "drives": drives,
            "state_signature": self.state_signature(obs)
        }
        return features

    # ------------------------------------------------------------------
    # Event Extraction
    # ------------------------------------------------------------------
    def _derive_events(self):
        if len(self.obs_history) < 2:
            return
        cur = self.obs_history[-1]
        prev = self.obs_history[-2]

        # Drive change events
        for k in self.drive_keys:
            if k in cur and k in prev:
                diff = cur[k] - prev[k]
                if abs(diff) > self.change_eps:
                    evt_type = "state_change"
                    # Dedicated resource events for energy
                    if k == "energy":
                        if diff >= self.energy_gain_threshold:
                            evt_type = "resource_gain"
                        elif diff <= self.energy_loss_threshold:
                            evt_type = "resource_loss"

                    event = {
                        "type": evt_type,
                        "drive": k,
                        "delta": diff,
                        "from": prev[k],
                        "to": cur[k],
                        "tick": cur.get("tick")
                    }
                    self._attach_cause(event)
                    self.events_buffer.append(event)

        # Goal contact event
        if cur.get("goal") and cur.get("agent_pos") == cur.get("goal"):
            goal_event = {
                "type": "goal_contact",
                "position": cur["agent_pos"],
                "tick": cur.get("tick")
            }
            self._attach_cause(goal_event)
            self.events_buffer.append(goal_event)

        # Harm / help (temperature extremes or social proximity)
        self._maybe_harm_help(prev, cur)

    def _maybe_harm_help(self, prev: Dict[str, Any], cur: Dict[str, Any]):
        # Temperature outside comfort band example (0.3,0.7)
        if "temperature" in cur:
            temp = cur["temperature"]
            prev_temp = prev.get("temperature", temp)
            if temp < 0.25 or temp > 0.75:
                event = {
                    "type": "harm",
                    "channel": "temperature",
                    "value": temp,
                    "tick": cur.get("tick")
                }
                self._attach_cause(event)
                self.events_buffer.append(event)
            elif (prev_temp < 0.3 or prev_temp > 0.7) and (0.3 <= temp <= 0.7):
                event = {
                    "type": "help",
                    "channel": "temperature",
                    "value": temp,
                    "tick": cur.get("tick")
                }
                self._attach_cause(event)
                self.events_buffer.append(event)

        # Social proximity improvements / deterioration
        if "social_proximity" in cur and "social_proximity" in prev:
            diff = cur["social_proximity"] - prev["social_proximity"]
            if diff > 0.05:
                event = {
                    "type": "help",
                    "channel": "social",
                    "delta": diff,
                    "tick": cur.get("tick")
                }
                self._attach_cause(event)
                self.events_buffer.append(event)
            elif diff < -0.05:
                event = {
                    "type": "harm",
                    "channel": "social",
                    "delta": diff,
                    "tick": cur.get("tick")
                }
                self._attach_cause(event)
                self.events_buffer.append(event)

    # ------------------------------------------------------------------
    # Causal Attribution (Rule-Based)
    # ------------------------------------------------------------------
    def _attach_cause(self, event: Dict[str, Any]):
        """
        If the last action plausibly caused the event, annotate 'cause'.
        Simple adjacency + schema matching.
        """
        if not self.action_history:
            return
        last_action = self.action_history[-1]

        # Mapping rules
        if event["type"] in ("resource_gain", "resource_loss") and last_action.get("type") in ("move", "gather", "consume"):
            event["cause"] = last_action.get("type")
        elif event["type"] == "goal_contact" and last_action.get("type") == "move":
            event["cause"] = "move_to_goal"
        elif event["type"] in ("harm", "help"):
            # If movement could have altered environmental condition
            if last_action.get("type") == "move":
                event["cause"] = "movement_environment_change"
        elif event["type"] == "state_change" and last_action.get("type") == "move":
            event["cause"] = "movement_drive_effect"

        # If still no cause, label 'unknown' for transparency
        event.setdefault("cause", "unknown")

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------
    def _prev_obs(self) -> Optional[Dict[str, Any]]:
        if len(self.obs_history) < 2:
            return None
        return self.obs_history[-2]

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def state_signature(self, obs: Dict[str, Any]) -> str:
        """
        Produce a compact hash signature of salient discrete state features
        used for reflex patches & surprise region clustering.
        """
        agent_pos = obs.get("agent_pos", (0, 0))
        goal = obs.get("goal")
        drives = [f"{k}:{round(obs.get(k, 0.0),3)}" for k in self.drive_keys if k in obs]
        base = f"{agent_pos}|{goal}|{'|'.join(drives)}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------
# Basic Self Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = Eventifier()
    p.feed_action({"type": "move", "dx": 1, "dy": 0})
    p.observe({"agent_pos": (0,0), "goal": (1,0), "energy": 0.5, "temperature": 0.6, "social_proximity": 0.2})
    p.feed_action({"type": "move", "dx": 1, "dy": 0})
    p.observe({"agent_pos": (1,0), "goal": (1,0), "energy": 0.53, "temperature": 0.8, "social_proximity": 0.28})
    events = p.extract_events()
    print("Events:", events)