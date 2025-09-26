"""
Social Model / Promise Management

Responsibilities:
- Register, track, and enforce social promises
- Provide constraint signals to planner (avoid violations)
- Support penalties on breach / expiry
- Log structural changes via optional callback

Promise Structure:
{
  "id": str,
  "condition": str,          # e.g. "avoid:(x,y)" or "deliver:item42"
  "behavior": str,           # free-form description of expected behavior
  "expiry": int,             # tick number when promise expires
  "penalty": str,            # spec e.g. "cost:5" or "principle:keep_promises"
  "created_at": int,
  "status": "active" | "fulfilled" | "breached" | "expired",
  "breach_tick": int | None,
  "fulfill_tick": int | None,
  "override_proof": dict | None
}

Penalty Semantics (lightweight for MVP):
- "cost:X"  => numeric planner penalty X
- "principle:P" => elevate principle P (placeholder: exposed as structural flag)
- Unknown pattern => treated as neutral (logged)

APIs:
register(condition, behavior, expiry, penalty) -> promise_id
update_tick(current_tick) : expire promises
mark_fulfilled(promise_id, tick)
breach(promise_id, tick, reason)
override(promise_id, proof_dict)

Planner Integration:
- is_promise_violation(node) style checks parse "avoid:(x,y)"
- Additional condition types can be added with pattern handlers.

Serialization:
to_dict() / from_dict()

Safety:
No eval beyond controlled pattern parsing `(x,y)` tuples.

"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Callable, List
import re
import itertools


class PromiseBook:
    def __init__(self, change_log_cb: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.promises: Dict[str, Dict[str, Any]] = {}
        self._id_counter = itertools.count(1)
        self._change_log_cb = change_log_cb

    # ------------------------------------------------------------------
    # Registration & Lifecycle
    # ------------------------------------------------------------------
    def register(self, condition: str, behavior: str, expiry: int, penalty: str) -> str:
        pid = f"P{next(self._id_counter):04d}"
        prom = {
            "id": pid,
            "condition": condition,
            "behavior": behavior,
            "expiry": int(expiry),
            "penalty": penalty,
            "created_at": 0,  # agent supplies current tick externally if desired
            "status": "active",
            "breach_tick": None,
            "fulfill_tick": None,
            "override_proof": None
        }
        self.promises[pid] = prom
        self._emit({"type": "promise_registered", "promise": prom.copy()})
        return pid

    def update_tick(self, current_tick: int):
        for prom in self.promises.values():
            if prom["status"] == "active" and current_tick >= prom["expiry"]:
                prom["status"] = "expired"
                self._emit({"type": "promise_expired", "id": prom["id"], "tick": current_tick})

    def mark_fulfilled(self, promise_id: str, tick: int):
        p = self.promises.get(promise_id)
        if not p or p["status"] != "active":
            return
        p["status"] = "fulfilled"
        p["fulfill_tick"] = tick
        self._emit({"type": "promise_fulfilled", "id": promise_id, "tick": tick})

    def breach(self, promise_id: str, tick: int, reason: str):
        p = self.promises.get(promise_id)
        if not p or p["status"] not in ("active",):
            return
        p["status"] = "breached"
        p["breach_tick"] = tick
        p["breach_reason"] = reason
        self._emit({"type": "promise_breached", "id": promise_id, "tick": tick, "reason": reason})

    def override(self, promise_id: str, proof: Dict[str, Any]):
        """
        Override (suspend) an active promise with explicit proof:
        proof = {
          "reason": str,
          "tradeoffs": list[str],
          "timestamp": int,
          "evidence": str
        }
        """
        required = ["reason", "tradeoffs", "timestamp", "evidence"]
        for k in required:
            if k not in proof:
                raise ValueError(f"Override proof missing field: {k}")
        p = self.promises.get(promise_id)
        if not p:
            raise ValueError("Unknown promise")
        p["override_proof"] = proof
        p["status"] = "breached"  # treated as breach but justified
        self._emit({"type": "promise_overridden", "id": promise_id, "proof": proof})

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def active_promises(self) -> List[Dict[str, Any]]:
        return [p for p in self.promises.values() if p["status"] == "active"]

    def get(self, promise_id: str) -> Optional[Dict[str, Any]]:
        return self.promises.get(promise_id)

    def penalty_value(self, promise_id: str) -> float:
        p = self.promises.get(promise_id)
        if not p:
            return 0.0
        pen = p["penalty"]
        if pen.startswith("cost:"):
            try:
                return float(pen.split("cost:")[1])
            except Exception:
                return 0.0
        return 0.0

    # ------------------------------------------------------------------
    # Constraint Helpers
    # ------------------------------------------------------------------
    _AVOID_RE = re.compile(r"^avoid:\(([-+]?\d+),\s*([-+]?\d+)\)$")

    def node_violates(self, node: Tuple[int, int]) -> bool:
        """
        Returns True if visiting node would breach any active avoid promise.
        """
        if not isinstance(node, tuple) or len(node) != 2:
            return False
        for p in self.active_promises():
            m = self._AVOID_RE.match(p["condition"])
            if m:
                tx, ty = int(m.group(1)), int(m.group(2))
                if node == (tx, ty):
                    return True
        return False

    def mark_breach_if_needed(self, node: Tuple[int, int], tick: int):
        for p in self.active_promises():
            m = self._AVOID_RE.match(p["condition"])
            if m:
                tx, ty = int(m.group(1)), int(m.group(2))
                if node == (tx, ty):
                    self.breach(p["id"], tick, reason=f"Entered forbidden node {node}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"promises": self.promises}

    def from_dict(self, data: Dict[str, Any]):
        block = data.get("promises", {})
        if isinstance(block, dict):
            self.promises = block

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _emit(self, record: Dict[str, Any]):
        if self._change_log_cb:
            try:
                self._change_log_cb(record)
            except Exception:
                pass


# ----------------------------------------------------------------------
# Basic self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pb = PromiseBook()
    pid = pb.register("avoid:(3,4)", "Do not step on tile (3,4)", expiry=50, penalty="cost:7")
    print("Registered:", pid, pb.promises[pid])
    print("Violation at (3,4)?", pb.node_violates((3, 4)))
    print("Violation at (2,4)?", pb.node_violates((2, 4)))
    pb.mark_breach_if_needed((3, 4), tick=10)
    print("After breach status:", pb.promises[pid]["status"])