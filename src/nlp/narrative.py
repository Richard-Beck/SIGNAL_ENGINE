# src/nlp/narrative.py

import uuid
import copy
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class Narrative:
    """
    Represents a market narrative with immutable core attributes.
    Lifecycle and performance attributes are mutable. New versions are
    created by 'evolving' an existing narrative.
    """
    def __init__(self,
                 narrative_id: str,
                 narrative_title: str,
                 narrative_description: str,
                 affected_tickers: List[Dict],
                 key_events: List[Dict],
                 parent_id: Optional[str] = None,
                 status: str = "CANDIDATE",
                 created_at: Optional[datetime] = None):

        # --- Immutable Core Attributes ---
        self.narrative_id = narrative_id
        self.parent_id = parent_id
        self.narrative_title = narrative_title
        self.narrative_description = narrative_description
        self.affected_tickers = affected_tickers
        self.key_events = key_events

        # --- Mutable Lifecycle & Performance Attributes ---
        self.status = status
        self.created_at = created_at or datetime.now()
        self.last_updated_at = self.created_at
        self.last_matched_at: Optional[datetime] = None
        self.hit_count = 0
        self.performance_score = 0.0

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'Narrative':
        """
        Factory method to create a brand new Narrative from a JSON object.
        These are "genesis" narratives with no parent.
        """
        return cls(
            narrative_id=json_data.get('narrative_id', f"narrative_{uuid.uuid4().hex[:8]}"),
            narrative_title=json_data['narrative_title'],
            narrative_description=json_data['narrative_description'],
            affected_tickers=json_data.get('affected_tickers', []),
            key_events=json_data.get('key_events', [])
        )

    @classmethod
    def evolve(cls, parent_narrative: 'Narrative', changes: Dict[str, Any]) -> 'Narrative':
        """
        Creates a new, evolved narrative from a parent. This is the primary
        way to modify a narrative's core attributes (e.g., after an LLM merge).
        """
        new_data = {
            "narrative_title": parent_narrative.narrative_title,
            "narrative_description": parent_narrative.narrative_description,
            "affected_tickers": copy.deepcopy(parent_narrative.affected_tickers),
            "key_events": copy.deepcopy(parent_narrative.key_events)
        }
        new_data.update(changes)

        return cls(
            narrative_id=f"narrative_{uuid.uuid4().hex[:8]}",
            parent_id=parent_narrative.narrative_id,
            narrative_title=new_data['narrative_title'],
            narrative_description=new_data['narrative_description'],
            affected_tickers=new_data['affected_tickers'],
            key_events=new_data['key_events']
        )

    # --- Public Methods for the Manager to Call ---
    def record_match(self, match_timestamp: datetime, matched_text: str, similarity_score: float) -> Dict[str, Any]:
        """
        Updates the narrative's internal state and score after a successful match.
        
        This method is now also responsible for returning a dictionary of the
        match details, which will be used to create the MatchEvent.

        Returns:
            A dictionary of primitive types representing the match event data.
        """
        # --- 1. Update Internal State ---
        self.hit_count += 1
        self.last_matched_at = match_timestamp
        self.last_updated_at = match_timestamp

        # --- 2. Return the Event Data Payload ---
        return {
            "narrative_id": self.narrative_id,
            "narrative_status": self.status,
            "narrative_title": self.narrative_title,
            "affected_tickers": self.affected_tickers,
            "match_timestamp": self.last_matched_at,
            "matched_text": matched_text,
            "similarity_score": similarity_score
        }

    def change_status(self, new_status: str):
        """Changes the narrative's lifecycle status."""
        self.status = new_status
        self.last_updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Narrative object back into a dictionary for serialization.
        """
        return {
            "narrative_id": self.narrative_id,
            "parent_id": self.parent_id,
            "narrative_title": self.narrative_title,
            "narrative_description": self.narrative_description,
            "affected_tickers": self.affected_tickers,
            "key_events": self.key_events,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "last_matched_at": self.last_matched_at.isoformat() if self.last_matched_at else None,
            "hit_count": self.hit_count,
            "performance_score": self.performance_score
        }

    def __repr__(self) -> str:
        return f"Narrative(id='{self.narrative_id}', parent='{self.parent_id}', status='{self.status}')"


# ==============================================================================
# Main execution block for testing the new Narrative class structure
# ==============================================================================
if __name__ == "__main__":
    print("--- 🧪 Testing the new Narrative Class ---")

    # --- Test Case 1: Create a "genesis" narrative from JSON ---
    print("\n--- Test Case 1: Create a new narrative from JSON ---")
    genesis_json_string = """
    {
      "narrative_title": "Initial AI Chip Demand",
      "narrative_description": "Early signs show demand for AI chips is growing."
    }
    """
    try:
        narrative_data = json.loads(genesis_json_string)
        genesis_narrative = Narrative.from_json(narrative_data)

        print(f"✅ Successfully created genesis narrative: {genesis_narrative}")
        assert genesis_narrative.parent_id is None
        assert genesis_narrative.status == "CANDIDATE"
        print("   - Asserted parent_id is None and status is CANDIDATE.")

    except Exception as e:
        print(f"❌ Test Case 1 Failed: {e}")


    # --- Test Case 2: Evolve an existing narrative ---
    print("\n--- Test Case 2: Evolve a narrative with new data ---")
    try:
        # Define the changes, for example, from an LLM merge
        evolution_changes = {
            "narrative_title": "Surging AI Chip Demand",
            "narrative_description": "Surging demand for AI chips is now causing supply chain bottlenecks.",
            "affected_tickers": [{"ticker": "NVDA", "position_if_true": "LONG"}]
        }

        # Create the new, evolved version
        evolved_narrative = Narrative.evolve(
            parent_narrative=genesis_narrative,
            changes=evolution_changes
        )

        print(f"✅ Successfully evolved narrative: {evolved_narrative}")
        assert evolved_narrative.narrative_id != genesis_narrative.narrative_id
        assert evolved_narrative.parent_id == genesis_narrative.narrative_id
        assert evolved_narrative.narrative_title == "Surging AI Chip Demand"
        assert len(evolved_narrative.affected_tickers) == 1
        print(f"   - Asserted new ID, correct parent_id, and updated title.")

    except Exception as e:
        print(f"❌ Test Case 2 Failed: {e}")

    # --- Test Case 3: Test mutable lifecycle methods ---
    print("\n--- Test Case 3: Update lifecycle attributes ---")
    try:
        # Start with a fresh narrative object
        test_narrative = Narrative.from_json(json.loads(genesis_json_string))
        print(f"   - Initial state: score={test_narrative.performance_score}, status='{test_narrative.status}'")

        # Simulate a match
        test_narrative.record_match(match_similarity=0.92)
        assert test_narrative.hit_count == 1
        assert test_narrative.performance_score == 0.92
        print(f"   - After match: score={test_narrative.performance_score}, hits={test_narrative.hit_count}")

        # Change its status
        test_narrative.change_status("LIVE")
        assert test_narrative.status == "LIVE"
        print(f"   - After promotion: status='{test_narrative.status}'")
        print("✅ Successfully updated mutable attributes.")

    except Exception as e:
        print(f"❌ Test Case 3 Failed: {e}")
    
    # --- Test Case 4: Real narrative ---
    print("\n--- Test Case 4: real narrative ---")
    real_narrative = """
    {
      "narrative_id": "tech_reshoring_tariff_mitigation",
      "narrative_title": "Tech Giants Invest in Domestic Manufacturing to Preempt Tariffs",
      "narrative_description": "Companies like Apple are making significant domestic investments to avoid potential tariffs on imported electronics, securing supply chains and improving cost efficiency. This could enhance margins and reduce geopolitical risks for affected firms.",
      "affected_tickers": [
        {
          "ticker": "AAPL",
          "position_if_true": "LONG"
        }
      ],
      "key_events": [
        {
          "event_id": "tariff_imposition",
          "event_type": "CONFIRMING",
          "event_description": "U.S. government imposes 15% tariffs on imported smartphones and consumer electronics effective Q1 2026.",
          "paraphrased_statements": [
            "Breaking: White House finalizes 15% tariff on foreign-made smartphones starting next quarter.",
            "Analysts note Apple’s U.S. production shields it from billions in new import duties.",
            "Trade Desk Alert: Apple shares rise pre-market as tariff announcement validates domestic strategy.",
            "Sector Update: Tech stocks diverge; Apple gains on tariff news while overseas suppliers slump.",
            "Political Insider: Bipartisan support for tariffs accelerates reshoring trends in tech manufacturing.",
            "Market Reaction: AAPL up 3% as tariffs incentivize local production expansion.",
            "Fed Watch: Tariffs may delay rate cuts, but Apple’s margins could cushion EPS impact.",
            "Trader Take: Long AAPL as tariffs increase cost advantages for domestic manufacturers.",
            "Corporate Strategy: Apple’s $100B U.S. investment now seen as prescient risk mitigation.",
            "Supply Chain Digest: Foxconn shifts iPhone assembly to Arizona amid tariff pressures.",
            "Wall Street View: Morgan Stanley upgrades AAPL, citing tariff-driven EPS upside.",
            "CNBC Guest: ‘Apple’s U.S. factories could absorb 40% of global iPhone output by 2027.’",
            "Bloomberg Radio: ‘Domestic tech manufacturing now a strategic priority post-tariff announcement.’",
            "Retail Investor Note: Tariff exemptions for U.S.-made goods favor Apple’s pricing power.",
            "Policy Brief: Commerce Secretary cites national security basis for electronics tariffs.",
            "Tech Conference Call: Tim Cook emphasizes tariff resilience in supply chain webinar.",
            "Earnings Preview: Analysts project Apple to raise guidance on tariff hedge tailwinds.",
            "Short Seller Warning: Bears cite overvaluation, but tariff benefits may sustain rally.",
            "Institutional Flow: Hedge funds pile into AAPL calls post-tariff headlines.",
            "Morning Report: Asian suppliers face headwinds; Apple’s onshoring offsets risks."
          ]
        },
        {
          "event_id": "tariff_rollback",
          "event_type": "REFUTING",
          "event_description": "U.S. suspends proposed consumer electronics tariffs amid international trade negotiations.",
          "paraphrased_statements": [
            "Exclusive: Biden administration pauses tariffs on Chinese electronics after diplomatic breakthrough.",
            "Sector Turmoil: Apple dips 4% as tariff reversal undermines reshoring thesis.",
            "Trade Update: US-China deal removes iPhone tariffs, easing cost pressures for importers.",
            "Analyst Take: Apple’s domestic investments now look overcautious post-tariff suspension.",
            "Breaking News: Treasury Secretary Yellen announces tariff deferral to ease inflation.",
            "Market Response: AAPL underperforms as tariff tailwind evaporates overnight.",
            "Policy Shift: Bipartisan lawmakers push to repeal tariffs ahead of midterm elections.",
            "CNBC Guest: ‘Tariff rollback exposes Apple’s costly bet on redundant U.S. capacity.’",
            "Supply Chain Weekly: Foxconn reconsiders Arizona expansion after policy reversal.",
            "Wall Street Memo: Goldman Sachs downgrades AAPL on reduced tariff hedge value.",
            "Retail Investor Alert: Options volume spikes as traders price out tariff premiums.",
            "ECON 101: Tariff suspension signals cooling trade tensions, reshuffles tech valuations.",
            "Corporate Brief: Apple issues statement supporting ‘free and fair global trade policies.’",
            "Futures Watch: Nasdaq futures slip as tariff-sensitive tech stocks face reassessment.",
            "Hedge Fund Diary: Contrarian funds short AAPL, betting on tariff policy overreach.",
            "Political Analysis: Tariff suspension seen as concession to secure semiconductor accord.",
            "Earnings Impact: UBS cuts Apple’s FY26 EPS estimates by 8% post-tariff news.",
            "Tech Today: Asian suppliers surge while domestic manufacturers face profit warnings.",
            "IPO Radar: Tariff reversal revives interest in Chinese smartphone component makers.",
            "Closing Bell: AAPL gives up weekly gains as investors recalibrate tariff risks."
          ]
        }
      ]
    }
    """
    try:
        narrative_data = json.loads(real_narrative)
        narrative_obj = Narrative.from_json(narrative_data)
        print(f"✅ Successfully created narrative: {narrative_obj}")
        print(f"   - Status: {narrative_obj.status}")
        print(f"   - Created At (as datetime obj): {narrative_obj.created_at}")
        print(f"   - Affected Tickers: {narrative_obj.affected_tickers[0]['ticker']}")

        # Test serialization back to a dictionary
        serialized_data = narrative_obj.to_dict()
        print("\n✅ Successfully serialized back to dictionary:")
        print(json.dumps(serialized_data, indent=2))
    except Exception as e:
        print(f"❌ Test Case 4 Failed with an unexpected error: {e}")