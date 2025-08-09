# narrative_eval.py
from typing import Any, Dict
from src.events.event_bus import event_bus, Event

class NarrativeEvaluator:
    """Stateless evaluator: on every match, emit a +1 score delta."""

    def __init__(self, bus=event_bus):
        self.bus = bus

    def register(self):
        self.bus.subscribe("NARRATIVE_MATCH", self._on_match)
        print("✅ NarrativeEvaluator subscribed to 'NARRATIVE_MATCH'")

    # EventBus passes the whole Event object
    def _on_match(self, event: Event):
        payload: Dict[str, Any] = event.payload or {}
        nid = payload.get("narrative_id")
        if not nid:
            return
        # publish a simple score delta; manager will apply it
        self.bus.publish(Event(
            "NARRATIVE_SCORE_DELTA",
            {"narrative_id": nid, "delta": 1}
        ))
