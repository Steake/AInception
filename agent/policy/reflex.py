"""
Reflex Layer

Purpose:
Fast, low-cost safeguards that can preempt deliberative planning when
simple, high-certainty conditions are detected.

Design Goals (MVP):
- Zero external dependencies
- Deterministic rule ordering (first matching rule fires)
- Clear justification tags for journal / auditing
- Easily extendable with add_rule()

Input:
maybe_act(features: dict, drive_errors: dict | None) where features originates
from Eventifier._build_features plus optional injected drive errors.

Returned Action Schema (if a reflex fires):
{
  "type": "noop" | "move" | ...,
  "reflex": True,
  "reflex_rule": str,
  "explanation": str
}

If no rule triggers, returns None (planner proceeds).

Current Minimal Rules Implemented:
1. Temperature Extreme Pause:
   If temperature drive present and outside [0.2, 0.8] band -> noop (avoid exacerbation).
2. Energy Conservation Idle:
   If energy deficit (error < -energy_low_margin) and distance_to_goal is large
   -> noop (conserve until planner can pick path).
3. Social Stabilization:
   If recent large negative social proximity delta (features may supply future extension) – (placeholder stub, inactive now).

Rule Evaluation Order:
The order they are stored in self._rules (append sequence).

Configuration (inline constants for MVP):
ENERGY_LOW_MARGIN = 0.15   (energy below setpoint by more than this triggers conservation)
GOAL_DISTANCE_CONSERVE_THRESHOLD = 4

Extension:
add_rule(name: str, predicate: Callable, action_builder: Callable)

Safety:
Each rule returns an action dict; we clamp to required keys and include a justification
summary for logging transparency.

"""

from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional, Tuple


class ReflexLayer:
    def __init__(
        self,
        energy_low_margin: float = 0.15,
        goal_distance_conserve_threshold: int = 4,
        temperature_low: float = 0.2,
        temperature_high: float = 0.8
    ):
        self.energy_low_margin = energy_low_margin
        self.goal_distance_conserve_threshold = goal_distance_conserve_threshold
        self.temperature_low = temperature_low
        self.temperature_high = temperature_high

        # Internal rule registry: list of (name, predicate, builder)
        self._rules: List[Tuple[str, Callable[[Dict[str, Any], Optional[Dict[str, float]]], bool], Callable[[Dict[str, Any], Optional[Dict[str, float]]], Dict[str, Any]]]] = []

        # Register default rules (order matters)
        self._register_default_rules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def maybe_act(self, features: Dict[str, Any], drive_errors: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """
        Iterate rules in order; return first action produced or None.
        """
        for name, predicate, builder in self._rules:
            try:
                if predicate(features, drive_errors):
                    action = builder(features, drive_errors)
                    # Normalize & annotate
                    return self._finalize_action(action, name)
            except Exception:
                # Fail-safe: ignore faulty rule
                continue
        return None

    def add_rule(
        self,
        name: str,
        predicate: Callable[[Dict[str, Any], Optional[Dict[str, float]]], bool],
        action_builder: Callable[[Dict[str, Any], Optional[Dict[str, float]]], Dict[str, Any]],
        prepend: bool = False
    ):
        entry = (name, predicate, action_builder)
        if prepend:
            self._rules.insert(0, entry)
        else:
            self._rules.append(entry)

    # ------------------------------------------------------------------
    # Internal: Default Rules
    # ------------------------------------------------------------------
    def _register_default_rules(self):
        # Rule 1: Temperature extreme pause
        def temp_extreme_pred(features: Dict[str, Any], _de: Optional[Dict[str, float]]) -> bool:
            drives = features.get("drives") or {}
            t = drives.get("temperature")
            if t is None:
                return False
            return t < self.temperature_low or t > self.temperature_high

        def temp_extreme_action(_features: Dict[str, Any], _de: Optional[Dict[str, float]]) -> Dict[str, Any]:
            return {
                "type": "noop",
                "explanation": "Temperature outside comfort band; pausing to avoid harm amplification."
            }

        self.add_rule("temperature_extreme_pause", temp_extreme_pred, temp_extreme_action)

        # Rule 2: Energy conservation idle
        def energy_conserve_pred(features: Dict[str, Any], drive_errors: Optional[Dict[str, float]]) -> bool:
            if not drive_errors:
                return False
            e_err = drive_errors.get("energy")
            if e_err is None:
                return False
            # error = current - setpoint. Negative large means deficit.
            if e_err >= -self.energy_low_margin:
                return False
            dist = features.get("distance_to_goal")
            if dist is None:
                return True  # no goal yet; still conserve
            return dist >= self.goal_distance_conserve_threshold

        def energy_conserve_action(features: Dict[str, Any], drive_errors: Optional[Dict[str, float]]) -> Dict[str, Any]:
            return {
                "type": "noop",
                "explanation": "Energy deficit detected; conserving before long traversal."
            }

        self.add_rule("energy_conservation_idle", energy_conserve_pred, energy_conserve_action)

        # Placeholder Rule 3 (inactive skeleton for social stabilization)
        def social_stabilize_pred(_f, _d):  # Always False until we wire social delta signals
            return False

        def social_stabilize_action(_f, _d):
            return {
                "type": "noop",
                "explanation": "Social stabilization pause."
            }

        self.add_rule("social_stabilization_pause", social_stabilize_pred, social_stabilize_action)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _finalize_action(self, action: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
        base = {
            "type": action.get("type", "noop"),
            "reflex": True,
            "reflex_rule": rule_name,
            "explanation": action.get("explanation", f"Reflex rule {rule_name} fired.")
        }
        # Merge any additional fields
        for k, v in action.items():
            if k not in base:
                base[k] = v
        return base


# ----------------------------------------------------------------------
# Basic Self-Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rl = ReflexLayer()
    # Simulate features
    features_hot = {
        "drives": {"temperature": 0.85, "energy": 0.5},
        "distance_to_goal": 6
    }
    features_low_energy = {
        "drives": {"temperature": 0.5, "energy": 0.45},
        "distance_to_goal": 7
    }
    de = {"energy": -0.25, "temperature": 0.0}

    print("Hot:", rl.maybe_act(features_hot, de))
    print("Low energy:", rl.maybe_act(features_low_energy, de))
    print("No trigger:", rl.maybe_act({"drives": {"temperature": 0.5, "energy": 0.7}, "distance_to_goal": 2}, {"energy": -0.05}))