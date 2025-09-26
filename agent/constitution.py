"""
Constitution Module

Responsibilities:
- Maintain ranked list of principles (higher priority = lower numeric rank value)
- Provide read API: ranking(), top(n)
- Enforce re-ranking only with supplied proof object
- Log each change (optionally via injected callback)
- Offer evaluation hook for planner/reflex to check candidate actions

Principle Representation:
Each principle stored with:
{
  "name": str,
  "rank": int,
  "description": str,
  "active": bool,
  "created_at": float,
  "last_updated": float
}

Re-Ranking:
set_ranking(new_order, proof)
- new_order: list of principle names in desired priority order (index = rank)
- proof: {
    "reason": str,
    "tradeoffs": list[str],
    "timestamp": int,
    "evidence": str,
    "affected_principles": list[str]
  }
All fields required for acceptance.
Reject if:
- Missing names
- Duplicates
- Unknown principle
- Proof incomplete

Evaluation Hook:
evaluate(action_context) returns {
  "violations": list[str],
  "checked": top_k_principles
}
Currently only structural; real rules can be injected externally.

Persistence:
Simple to_dict / from_dict for agent save/load.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
import time


@dataclass
class Principle:
    name: str
    rank: int
    description: str = ""
    active: bool = True
    created_at: float = None
    last_updated: float = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Principle":
        p = Principle(
            name=d["name"],
            rank=int(d["rank"]),
            description=d.get("description", ""),
            active=bool(d.get("active", True)),
            created_at=d.get("created_at"),
            last_updated=d.get("last_updated")
        )
        return p


class Constitution:
    def __init__(
        self,
        principles_spec: List[Any],
        change_log_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        principles_spec may be:
        - list[str] (names only) OR
        - list[dict] each with keys {name, description?, rank?}

        Ranks auto-assigned sequentially if absent (order given).
        """
        self._principles: Dict[str, Principle] = {}
        self._change_log_cb = change_log_cb
        now = time.time()
        if not principles_spec:
            principles_spec = [
                {"name": "do_not_harm", "description": "Avoid entering danger / forbidden zones."},
                {"name": "keep_promises", "description": "Honor registered social promises."},
                {"name": "conserve_energy", "description": "Prefer actions that maintain energy near setpoint."}
            ]
        # Normalize input
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(principles_spec):
            if isinstance(item, str):
                normalized.append({"name": item, "rank": idx})
            elif isinstance(item, dict):
                cp = dict(item)
                cp.setdefault("rank", idx)
                normalized.append(cp)
        # Build
        for spec in normalized:
            p = Principle(
                name=spec["name"],
                rank=int(spec["rank"]),
                description=spec.get("description", ""),
                active=bool(spec.get("active", True)),
                created_at=now,
                last_updated=now
            )
            self._principles[p.name] = p
        # Ensure contiguous ranks
        self._reassign_compact_ranks()

    # ------------------------------------------------------------------
    # Ranking APIs
    # ------------------------------------------------------------------
    def ranking(self) -> List[tuple[int, str]]:
        """Return list of (rank, principle_name) sorted by rank ascending."""
        return sorted([(p.rank, p.name) for p in self._principles.values()], key=lambda x: x[0])

    def top(self, k: int = 3) -> List[str]:
        return [name for _, name in self.ranking()[:k]]

    def get(self, name: str) -> Optional[Principle]:
        return self._principles.get(name)

    def principles_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._principles.items()}

    # ------------------------------------------------------------------
    # Modification / Re-Ranking
    # ------------------------------------------------------------------
    def set_ranking(self, new_order: List[str], proof: Dict[str, Any]):
        """
        Apply new total ordering of principles (index = rank).
        Requires proof with mandatory fields for audit.
        """
        self._validate_new_order(new_order)
        self._validate_proof(proof, new_order)
        # Apply
        now = time.time()
        for idx, name in enumerate(new_order):
            p = self._principles[name]
            if p.rank != idx:
                p.rank = idx
                p.last_updated = now
        self._reassign_compact_ranks()
        self._emit_change({
            "type": "re_rank",
            "timestamp": proof.get("timestamp", now),
            "reason": proof["reason"],
            "tradeoffs": proof["tradeoffs"],
            "evidence": proof["evidence"],
            "affected_principles": proof["affected_principles"],
            "new_order": new_order
        })

    def add_principle(self, name: str, description: str = "", proof: Optional[Dict[str, Any]] = None):
        if name in self._principles:
            return
        now = time.time()
        rank = len(self._principles)
        self._principles[name] = Principle(
            name=name,
            rank=rank,
            description=description,
            created_at=now,
            last_updated=now
        )
        self._emit_change({
            "type": "add_principle",
            "timestamp": now,
            "name": name,
            "description": description,
            "proof": proof
        })

    def deactivate(self, name: str, proof: Optional[Dict[str, Any]] = None):
        p = self._principles.get(name)
        if not p:
            return
        p.active = False
        p.last_updated = time.time()
        self._emit_change({
            "type": "deactivate_principle",
            "timestamp": p.last_updated,
            "name": name,
            "proof": proof
        })

    # ------------------------------------------------------------------
    # Evaluation Hook
    # ------------------------------------------------------------------
    def evaluate(self, action_ctx: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        """
        Basic structural evaluation stub.
        action_ctx may include:
          {
            "position": (x,y),
            "forbidden_tiles": set,
            "promises": list,
            "energy": float,
            ...
          }
        Returns:
          {
            "violations": list[str],
            "checked": list[str]
          }
        """
        checked = self.top(top_k)
        violations: List[str] = []

        pos = action_ctx.get("position")
        forbidden = action_ctx.get("forbidden_tiles") or set()
        if pos in forbidden and "do_not_harm" in self._principles:
            violations.append("do_not_harm")

        # Promise violation placeholder
        if action_ctx.get("promise_breach") and "keep_promises" in self._principles:
            violations.append("keep_promises")

        # Energy conservation heuristic: if energy would drop below 0.2 threshold
        projected_energy = action_ctx.get("projected_energy")
        if projected_energy is not None and projected_energy < 0.2 and "conserve_energy" in self._principles:
            violations.append("conserve_energy")

        return {
            "violations": violations,
            "checked": checked
        }

    # ------------------------------------------------------------------
    # Validation Helpers
    # ------------------------------------------------------------------
    def _validate_new_order(self, new_order: List[str]):
        if not new_order:
            raise ValueError("new_order must not be empty")
        existing = set(self._principles.keys())
        if set(new_order) != existing:
            missing = existing.difference(new_order)
            extra = set(new_order).difference(existing)
            raise ValueError(f"Invalid re-ranking: missing={missing} extra={extra}")
        if len(new_order) != len(set(new_order)):
            raise ValueError("Duplicate names in new_order")

    def _validate_proof(self, proof: Dict[str, Any], new_order: List[str]):
        required = ["reason", "tradeoffs", "timestamp", "evidence", "affected_principles"]
        for k in required:
            if k not in proof:
                raise ValueError(f"Proof missing field: {k}")
        if not isinstance(proof["tradeoffs"], list) or not proof["tradeoffs"]:
            raise ValueError("Proof.tradeoffs must be non-empty list")
        if not isinstance(proof["affected_principles"], list) or not proof["affected_principles"]:
            raise ValueError("Proof.affected_principles must be non-empty list")
        for n in proof["affected_principles"]:
            if n not in self._principles:
                raise ValueError(f"Proof references unknown principle: {n}")
        # Ensure affected subset appears in new_order
        for n in proof["affected_principles"]:
            if n not in new_order:
                raise ValueError(f"Affected principle {n} not in new_order")

    def _reassign_compact_ranks(self):
        ordered = sorted(self._principles.values(), key=lambda p: p.rank)
        for idx, p in enumerate(ordered):
            p.rank = idx

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "principles": {name: p.to_dict() for name, p in self._principles.items()}
        }

    def from_dict(self, data: Dict[str, Any]):
        block = data.get("principles", {})
        restored: Dict[str, Principle] = {}
        for name, spec in block.items():
            restored[name] = Principle.from_dict(spec)
        self._principles = restored
        self._reassign_compact_ranks()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _emit_change(self, record: Dict[str, Any]):
        if self._change_log_cb:
            try:
                self._change_log_cb(record)
            except Exception:
                pass


# ----------------------------------------------------------------------
# Basic self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    c = Constitution(["do_not_harm", "keep_promises", "conserve_energy"])
    print("Initial ranking:", c.ranking())
    proof = {
        "reason": "Observed repeated energy scarcity events",
        "tradeoffs": ["Slightly longer paths", "Potential delayed goal contact"],
        "timestamp": int(time.time()),
        "evidence": "Drive logs show energy error spikes at ticks 42-55",
        "affected_principles": ["conserve_energy", "keep_promises"]
    }
    c.set_ranking(["do_not_harm", "conserve_energy", "keep_promises"], proof)
    print("After re-rank:", c.ranking())
    eval_out = c.evaluate({"position": (1,1), "forbidden_tiles": {(1,1)}, "projected_energy": 0.15})
    print("Evaluation:", eval_out)