"""
Self-Journal Module

Responsibilities:
- Append structured events with causal/context data
- Generate natural language summary line (template or LLM stub)
- Maintain rolling buffers:
    * recent_events (last N)
    * lessons (surprise-tagged)
- Provide summarized views: last_100, top_lessons, daily_summary (placeholder)
- Persist to SQLite if connection provided (schema alignment with database/schema.py & models.py)

Event Input Expected Shape (flexible):
{
  "timestamp": int (agent tick),
  "type": str,
  "cause": str,
  ... arbitrary key-value pairs (delta, drive, etc.)
  "surprise_level": float (optional)
}

Action Justification Logging:
log_action(step_tick, action_dict, justification_dict, drive_errors_dict)

Natural Language:
- For MVP use template-based deterministic generation
- LLM hook: pass event dict to external generator if enabled

Lessons:
- Event qualifies as lesson if:
    * event["type"] in {"surprise_lesson"} OR
    * event.get("surprise_level", 0) > surprise_threshold
- Stored with minimal derived fields (affected_drive if present)

SQLite Persistence:
- Optional: pass sqlite3.Connection. If provided, insert events into events table
- Journal module does not own connection closing

Threading:
- Not thread safe by design (single-agent tick loop)

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import json
import time
import sqlite3


@dataclass
class JournalConfig:
    recent_capacity: int = 200
    lesson_capacity: int = 50
    surprise_threshold: float = 0.35
    top_lessons_k: int = 3
    use_llm: bool = False


class Journal:
    def __init__(
        self,
        config: JournalConfig | None = None,
        db_conn: Optional[sqlite3.Connection] = None,
        llm_fn: Optional[Callable[[Dict[str, Any]], str]] = None
    ):
        self.cfg = config or JournalConfig()
        self._recent: List[Dict[str, Any]] = []
        self._lessons: List[Dict[str, Any]] = []
        self._db = db_conn
        self._llm_fn = llm_fn if (self.cfg.use_llm and llm_fn) else None
        self._action_logs: List[Dict[str, Any]] = []  # lightweight in-memory store for step justifications

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def append_event(self, event: Dict[str, Any]):
        ev = dict(event)  # shallow copy
        ev.setdefault("timestamp", int(time.time()))
        if "text" not in ev:
            ev["text"] = self._render_text(ev)

        self._push_recent(ev)
        self._maybe_store_lesson(ev)
        self._persist_event(ev)

    def log_action(self, tick: int, action: Dict[str, Any], justification: Dict[str, Any], drive_errors: Dict[str, float]):
        """
        Structured action + justification logging separate from event stream.
        This can optionally be surfaced as events if needed.
        """
        record = {
            "timestamp": tick,
            "action": action,
            "justification": justification,
            "drive_errors": drive_errors
        }
        self._action_logs.append(record)
        # Optionally also append a condensed event
        action_event = {
            "timestamp": tick,
            "type": "action_decision",
            "cause": justification.get("fallback_reason") or "deliberation" if justification.get("type") != "reflex" else "reflex",
            "principles": justification.get("principles_checked"),
            "dominant_drive": justification.get("dominant_drive"),
            "surprise_flag": justification.get("surprise_flag", False)
        }
        self.append_event(action_event)

    def recent_events(self, n: int = 25) -> List[Dict[str, Any]]:
        return self._recent[-n:]

    def last_100(self) -> List[Dict[str, Any]]:
        return self._recent[-100:]

    def lessons(self) -> List[Dict[str, Any]]:
        return list(self._lessons)

    def top_lessons(self) -> List[Dict[str, Any]]:
        # naive: choose highest surprise_level present
        scored = [
            (e.get("surprise_level", 0.0), e)
            for e in self._lessons
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[: self.cfg.top_lessons_k]]

    def summary_snapshot(self) -> Dict[str, Any]:
        return {
            "recent_count": len(self._recent),
            "lessons_count": len(self._lessons),
            "top_lessons": [
                {
                    "timestamp": l["timestamp"],
                    "type": l["type"],
                    "surprise_level": l.get("surprise_level"),
                    "text": l.get("text")
                }
                for l in self.top_lessons()
            ],
        }

    # ------------------------------------------------------------------
    # Internal: Recent & Lessons
    # ------------------------------------------------------------------
    def _push_recent(self, ev: Dict[str, Any]):
        self._recent.append(ev)
        if len(self._recent) > self.cfg.recent_capacity:
            self._recent.pop(0)

    def _maybe_store_lesson(self, ev: Dict[str, Any]):
        s = ev.get("surprise_level", 0.0)
        if ev.get("type") == "surprise_lesson" or s > self.cfg.surprise_threshold:
            lesson = {
                "timestamp": ev["timestamp"],
                "type": ev.get("type", "lesson"),
                "surprise_level": s,
                "affected_drive": ev.get("drive") or ev.get("dominant_drive"),
                "text": ev.get("text"),
                "raw": {
                    k: ev[k]
                    for k in ev.keys()
                    if k not in {"raw"}  # avoid nesting
                }
            }
            self._lessons.append(lesson)
            if len(self._lessons) > self.cfg.lesson_capacity:
                self._lessons.pop(0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_event(self, ev: Dict[str, Any]):
        if not self._db:
            return
        try:
            cur = self._db.cursor()
            cur.execute(
                "INSERT INTO events(timestamp,event_type,delta,cause,consequence,surprise_level) VALUES (?,?,?,?,?,?)",
                (
                    ev.get("timestamp"),
                    ev.get("type"),
                    json.dumps({k: ev[k] for k in ev.keys() if k not in {"timestamp","type","cause","consequence","surprise_level","text"}}),
                    ev.get("cause"),
                    ev.get("consequence"),
                    ev.get("surprise_level", 0.0),
                ),
            )
            self._db.commit()
        except Exception:
            # Fail silently for now (could add logging hook)
            pass

    # ------------------------------------------------------------------
    # Text Rendering
    # ------------------------------------------------------------------
    def _render_text(self, ev: Dict[str, Any]) -> str:
        if self._llm_fn:
            try:
                return self._llm_fn(ev)
            except Exception:
                pass  # fallback to template

        etype = ev.get("type", "event")
        cause = ev.get("cause", "unknown")
        if etype == "resource_gain":
            return f"Gained resource: drive={ev.get('drive')} Δ={round(ev.get('delta',0),3)} (cause={cause})."
        if etype == "resource_loss":
            return f"Lost resource: drive={ev.get('drive')} Δ={round(ev.get('delta',0),3)} (cause={cause})."
        if etype == "goal_contact":
            return f"Reached goal at {ev.get('position')} (cause={cause})."
        if etype == "harm":
            return f"Harm signal channel={ev.get('channel')} val={ev.get('value',ev.get('delta'))} (cause={cause})."
        if etype == "help":
            return f"Help signal channel={ev.get('channel')} val={ev.get('value',ev.get('delta'))} (cause={cause})."
        if etype == "state_change":
            return f"State change drive={ev.get('drive')} from {round(ev.get('from',0),3)} to {round(ev.get('to',0),3)} (cause={cause})."
        if etype == "action_decision":
            return f"Action decision; top principles consulted={ev.get('principles')} cause={cause}."
        if etype == "surprise_lesson":
            return f"Surprise lesson magnitude={round(ev.get('surprise_level',0),3)} drive={ev.get('drive') or ev.get('dominant_drive')}."
        return f"{etype} recorded (cause={cause})."

    # ------------------------------------------------------------------
    # Utility / Maintenance
    # ------------------------------------------------------------------
    def clear(self):
        self._recent.clear()
        self._lessons.clear()
        self._action_logs.clear()

    def export_json(self) -> Dict[str, Any]:
        return {
            "recent": self._recent,
            "lessons": self._lessons,
            "actions": self._action_logs
        }


# ----------------------------------------------------------------------
# LLM Hook Example
# ----------------------------------------------------------------------
def simple_llm_stub(ev: Dict[str, Any]) -> str:
    """
    Placeholder 'LLM' generator. Deterministic stylized summarization.
    Replace with actual transformers model call externally.
    """
    base = f"[{ev.get('type','event')}]"
    detail = ""
    if "drive" in ev:
        detail += f" drive={ev['drive']}"
    if "delta" in ev:
        detail += f" Δ={round(ev['delta'],3)}"
    if "surprise_level" in ev and ev["surprise_level"] > 0:
        detail += f" surprise={round(ev['surprise_level'],3)}"
    return f"{base}{detail} cause={ev.get('cause','?')}"


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sqlite3
    conn = None
    j = Journal(JournalConfig(use_llm=False), db_conn=conn)
    j.append_event({"timestamp": 1, "type": "resource_gain", "drive": "energy", "delta": 0.05, "cause": "move"})
    j.append_event({"timestamp": 2, "type": "harm", "channel": "temperature", "value": 0.9})
    j.append_event({"timestamp": 3, "type": "surprise_lesson", "surprise_level": 0.6, "dominant_drive": "temperature"})
    j.log_action(4, {"type": "move", "dx": 1, "dy": 0}, {"principles_checked": ["do_not_harm","keep_promises","conserve_energy"]}, {"energy": -0.02})
    print("Recent:", len(j.recent_events()))
    print("Lessons:", j.top_lessons())
    print("Summary snapshot:", j.summary_snapshot())