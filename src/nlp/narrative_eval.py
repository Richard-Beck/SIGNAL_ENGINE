# narrative_eval.py
from typing import Any, Dict
from src.events.event_bus import event_bus, Event
import math 

class NarrativeEvaluator:
    """Stateless evaluator: on every match, emit a +1 score delta."""

    def __init__(self, bus=event_bus):
        self.bus = bus

    def register(self):
        self.bus.subscribe("NARRATIVE_MATCH", self._on_match)
        print("✅ NarrativeEvaluator subscribed to 'NARRATIVE_MATCH'")



    def _on_match(self, event: Event):
        """
        Calculates and applies a score delta based on an aggregated match event.
        """
        payload: Dict[str, Any] = event.payload or {}
        nid = payload.get("narrative_id")
        if not nid:
            return

        # --- 1. Check for Ticker Conflicts (Poor Narrative Design) ---
        # This checks if the narrative itself recommends both LONG and SHORT for the same ticker.
        # If so, the narrative is flawed, and we should not assign any score.
        tickers_seen = {}
        for item in payload.get('affected_tickers', []):
            ticker = item.get('ticker')
            position = item.get('position_if_true')
            if ticker in tickers_seen and tickers_seen[ticker] != position:
                print(f"  ! Ticker conflict in {nid} ({ticker}). Nullifying score.")
                # Do not reward poorly formed narratives.
                self.bus.publish(Event("NARRATIVE_SCORE_DELTA", {"narrative_id": nid, "delta": 0}))
                return
            tickers_seen[ticker] = position

        # --- 2. Calculate Raw Evidence Scores ---
        # Combine the quantity of matches with the quality (max similarity score).
        confirm_score = payload.get('confirming_matches', 0) * payload.get('max_confirming_score', 0.0)
        refute_score = payload.get('refuting_matches', 0) * payload.get('max_refuting_score', 0.0)
        
        # --- 3. Calculate Net Score and Penalize Mixed Signals ---
        # The net score is the difference between confirming and refuting evidence.
        net_score = confirm_score - refute_score

        # If we have both confirming AND refuting matches, the signal is unclear.
        # We penalize this uncertainty by halving the score's magnitude.
        if payload.get('confirming_matches', 0) > 0 and payload.get('refuting_matches', 0) > 0:
            net_score *= 0.5

        # --- 4. Normalize to [-1, 1] Range and Publish ---
        # A tunable factor to control how quickly the score saturates to +/- 1.
        SCALING_FACTOR = 0.5
        
        # Use the hyperbolic tangent function (tanh) to squash the score into the -1 to 1 range.
        final_delta = math.tanh(net_score * SCALING_FACTOR)
        
        self.bus.publish(Event(
            "NARRATIVE_SCORE_DELTA",
            {"narrative_id": nid, "delta": final_delta}
        ))
        print(f"  ↑ Score delta for {nid}: {final_delta:.4f} (Raw score: {net_score:.4f})")
