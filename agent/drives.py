"""
Homeostatic Drives Module

Responsibilities:
- Maintain continuous drives with setpoints
- Update drive values from sensed state each tick
- Compute per-drive error, weighted quadratic cost
- Provide reward contribution (negative of aggregated cost)
- Support serialization (save/load)
- Provide projection utility for planner heuristic (estimate effect of hypothetical delta)

Drives (initial):
  energy, temperature, social_proximity

Each drive has:
  name: str
  setpoint: float
  weight: float
  current: float
  min_val: float
  max_val: float
  decay_rate: optional (for autonomous drift each tick)

Reward Model:
  base_drive_cost = Σ weight_i * (current_i - setpoint_i)^2
  reward = -base_drive_cost - action_cost (action_cost supplied externally)

Normalization:
  Expose normalized errors ( (current - setpoint) / (max_val - min_val) ) for neural model
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Iterable
import json
import math


@dataclass
class Drive:
    name: str
    setpoint: float
    weight: float
    current: float
    min_val: float = 0.0
    max_val: float = 1.0
    decay_rate: float = 0.0  # passive drift per tick toward min_val if positive

    def clamp(self):
        if self.current < self.min_val:
            self.current = self.min_val
        elif self.current > self.max_val:
            self.current = self.max_val

    def error(self) -> float:
        return self.current - self.setpoint

    def normalized_error(self) -> float:
        rng = (self.max_val - self.min_val) or 1.0
        return (self.current - self.setpoint) / rng

    def cost(self) -> float:
        e = self.error()
        return self.weight * e * e


class DriveSystem:
    def __init__(self, drive_specs: Dict[str, Dict[str, Any]]):
        """
        drive_specs example:
        {
          "energy": {
             "setpoint": 0.7,
             "weight": 1.0,
             "initial": 0.7,
             "min": 0.0,
             "max": 1.0,
             "decay_rate": 0.01
          },
          ...
        }
        """
        self.drives: Dict[str, Drive] = {}
        for name, spec in drive_specs.items():
            drv = Drive(
                name=name,
                setpoint=float(spec["setpoint"]),
                weight=float(spec.get("weight", 1.0)),
                current=float(spec.get("initial", spec["setpoint"])),
                min_val=float(spec.get("min", 0.0)),
                max_val=float(spec.get("max", 1.0)),
                decay_rate=float(spec.get("decay_rate", 0.0))
            )
            drv.clamp()
            self.drives[name] = drv

    # ---------------------------------------------------------------------
    # Core Update
    # ---------------------------------------------------------------------
    def tick_decay(self):
        """Apply passive decay / drift per tick."""
        for d in self.drives.values():
            if d.decay_rate > 0.0:
                d.current -= d.decay_rate
                d.clamp()

    def ingest_observation(self, obs: Dict[str, Any]):
        """
        Update drives from observation if keys present.
        Example mapping: obs may contain energy, temperature, social_proximity
        """
        for k, v in obs.items():
            if k in self.drives and isinstance(v, (int, float)):
                self.drives[k].current = float(v)
                self.drives[k].clamp()

    # ---------------------------------------------------------------------
    # Metrics & Costs
    # ---------------------------------------------------------------------
    def drive_errors(self) -> Dict[str, float]:
        return {k: d.error() for k, d in self.drives.items()}

    def normalized_errors(self) -> Dict[str, float]:
        return {k: d.normalized_error() for k, d in self.drives.items()}

    def total_cost(self) -> float:
        return sum(d.cost() for d in self.drives.values())

    def reward(self, action_cost: float = 0.0) -> float:
        """
        Reward is negative of drive cost minus external action cost.
        """
        return -self.total_cost() - action_cost

    # ---------------------------------------------------------------------
    # Projection Utilities
    # ---------------------------------------------------------------------
    def project_with_deltas(self, deltas: Dict[str, float]) -> float:
        """
        Estimate future cost if specified delta adjustments applied.
        deltas: mapping drive_name -> additive delta to current value
        Returns projected total_cost.
        """
        total = 0.0
        for k, d in self.drives.items():
            val = d.current + deltas.get(k, 0.0)
            # clamp
            if val < d.min_val:
                val = d.min_val
            elif val > d.max_val:
                val = d.max_val
            e = (val - d.setpoint)
            total += d.weight * e * e
        return total

    # ---------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {k: asdict(d) for k, d in self.drives.items()}

    def from_dict(self, data: Dict[str, Any]):
        for k, spec in data.items():
            if k in self.drives:
                drv = self.drives[k]
                drv.current = float(spec.get("current", drv.current))
                drv.setpoint = float(spec.get("setpoint", drv.setpoint))
                drv.weight = float(spec.get("weight", drv.weight))
                drv.min_val = float(spec.get("min_val", drv.min_val))
                drv.max_val = float(spec.get("max_val", drv.max_val))
                drv.decay_rate = float(spec.get("decay_rate", drv.decay_rate))
                drv.clamp()

    def dumps(self) -> str:
        return json.dumps(self.to_dict())

    def loads(self, s: str):
        self.from_dict(json.loads(s))

    # ---------------------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------------------
    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            k: {
                "current": d.current,
                "setpoint": d.setpoint,
                "weight": d.weight,
                "error": d.error(),
                "normalized_error": d.normalized_error()
            } for k, d in self.drives.items()
        }

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------
    def largest_error(self) -> Optional[str]:
        if not self.drives:
            return None
        return max(self.drives.items(), key=lambda kv: abs(kv[1].error()))[0]

    def adjust_setpoint(self, name: str, new_setpoint: float):
        if name in self.drives:
            self.drives[name].setpoint = new_setpoint
            self.drives[name].clamp()

    def set_drive(self, name: str, value: float):
        if name in self.drives:
            d = self.drives[name]
            d.current = value
            d.clamp()


# -------------------------------------------------------------------------
# Helper factory
# -------------------------------------------------------------------------
def build_from_config(cfg: Dict[str, Any]) -> DriveSystem:
    """
    cfg expected shape:
    {
      "drives": {
         "energy": {...},
         "temperature": {...},
         "social_proximity": {...}
      }
    }
    """
    drives_cfg = cfg.get("drives", {})
    return DriveSystem(drives_cfg)


# -------------------------------------------------------------------------
# Basic self-test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    spec = {
        "energy": {"setpoint": 0.7, "weight": 1.0, "initial": 0.5, "decay_rate": 0.01},
        "temperature": {"setpoint": 0.5, "weight": 0.8, "initial": 0.6},
        "social_proximity": {"setpoint": 0.3, "weight": 0.5, "initial": 0.2}
    }
    ds = DriveSystem(spec)
    print("Initial summary:", ds.summary())
    ds.tick_decay()
    ds.ingest_observation({"energy": 0.65})
    print("Post update summary:", ds.summary())
    print("Reward (no action cost):", ds.reward())