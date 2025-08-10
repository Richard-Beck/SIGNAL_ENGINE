# src/nlp/narrative_manager.py

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import pickle
from src.events.event_bus import Event, event_bus
from src.nlp.narrative import Narrative
from src.nlp.narrative_eval import NarrativeEvaluator
import re
import sys
from contextlib import contextmanager
from tqdm import tqdm

@contextmanager
def suppress_output():
    """Suppress stdout/stderr (C-level and Python) temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class NarrativeManager:

    """
    Manages the entire lifecycle, state, and indexing of all narratives.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the NarrativeManager, loading the embedding model
        and initializing a real FAISS index.
        """
        print("--- 🚀 Initializing NarrativeManager---")
        self.config = config
        self.narratives: Dict[str, Narrative] = {}
        self.index_map: List[Dict[str, Any]] = []

        # Directly initialize the real models upon creation.
        self._initialize_models()
        
        # Initialize with an empty but valid index
        self._rebuild_index()
        print("✅ NarrativeManager Initialized.")

        self.evaluator = NarrativeEvaluator()
        self.evaluator.register()
        event_bus.subscribe("NARRATIVE_SCORE_DELTA", self._apply_score_delta)
        event_bus.subscribe("NARRATIVE_FILE_READY", self._on_narrative_file_ready)
    # --- Public Interfaces ---

    def add_narratives(self, new_narratives: List[Narrative]):
        """Public method to add new candidate narratives to the manager."""
        print(f"\n--- 📥 Adding {len(new_narratives)} new narratives ---")
        for narrative in new_narratives:
            if narrative.narrative_id not in self.narratives:
                self.narratives[narrative.narrative_id] = narrative
            else:
                print(f"  ! Skipped duplicate narrative ID: {narrative.narrative_id}")
        self._rebuild_index()

    def match_text_stream(self, text: str) -> List[Dict[str, Any]]:
        """
        Finds ALL matching narratives for the input text that exceed the
        similarity threshold.


        Args:
            text: The input text from the real-time stream.

        Returns:
            A list of all match event payloads, or an empty list if no matches found.
        """
        import faiss
        import numpy as np

        if self.index.ntotal == 0:
            return []

        with suppress_output():
            query_embedding = np.array(self.embedding_model.embed([text]), dtype='float32')
        
        faiss.normalize_L2(query_embedding)

        threshold = self.config.DEFAULT_SIMILARITY_THRESHOLD
        lims, distances, indices = self.index.range_search(query_embedding, thresh=threshold)

        # --- MODIFICATION START ---
        # The function now returns a list of payloads instead of publishing events.
        found_payloads = []
        
        for i in range(len(indices)):
            similarity_score = distances[i]
            match_index = indices[i]
            
            match_info = self.index_map[match_index]
            matched_narrative_id = match_info['narrative_id']
            matched_narrative = self.narratives[matched_narrative_id]

            # This call still updates the narrative's internal state (e.g., counters)
            # and conveniently returns a data payload.
            event_payload = matched_narrative.record_match(
                match_timestamp=datetime.now(),
                matched_text=match_info['statement_text'],
                similarity_score=similarity_score
            )
            #print(event_payload)

            # We collect the payload to be returned to the caller.
            found_payloads.append(event_payload)

        if found_payloads:
            print(f"  ✅ Found {len(found_payloads)} potential matches for aggregation.")

        return found_payloads


    def update_score(self, narrative_id: str, new_score: float):
        """Allows an external system to update a narrative's score."""
        if narrative_id in self.narratives:
            self.narratives[narrative_id].performance_score = new_score
            print(f"  ⬆️ Score updated for {narrative_id} to {new_score:.2f}")

    def load_narratives_from_folder(self, folder_path: str):
        """
        Loops over all .txt files in a directory and loads the narratives
        from each one using the private _load_narrative method.

        Args:
            folder_path: The full path to the directory containing narrative files.
        """
        print(f"\n--- 📂 Loading narratives from folder: {folder_path} ---")
        if not os.path.isdir(folder_path):
            print(f"  ! Error: Folder not found at {folder_path}")
            return

        # Use a list to collect all narratives from all files first
        all_narratives_to_add = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                # The helper function returns a list of narratives from a single file
                narratives_from_file = self._load_narrative(filepath)
                all_narratives_to_add.extend(narratives_from_file)
        
        # Add all newly created narrative objects to the manager at once.
        # The add_narratives method already handles duplicate checking.
        if all_narratives_to_add:
            self.add_narratives(all_narratives_to_add)
    
    def load_narrative_file(self, filepath: str) -> int:
        """
        Load a single narrative file and rebuild the FAISS index from cached embeddings.
        Only *new* narratives will compute embeddings (via compute_embeddings(force=False) in _rebuild_index).
        """
        import os
        if not os.path.isfile(filepath):
            print(f"  ! File not found: {filepath}")
            return 0

        new_list = self._load_narrative(filepath)
        if not new_list:
            print(f"  ! No narratives parsed from {os.path.basename(filepath)}")
            return 0

        # Add only unseen narratives
        fresh = [n for n in new_list if n.narrative_id not in self.narratives]
        for n in fresh:
            self.narratives[n.narrative_id] = n

        if fresh:
            # Rebuilds FAISS from cached vectors; computes embeddings only for missing ones
            self._rebuild_index()
            print(f"📥 Ingested {len(fresh)} new narratives from {os.path.basename(filepath)}")
        else:
            print(f"ℹ️ No new narratives in {os.path.basename(filepath)} (all IDs existed).")

        return len(fresh)


    # --- Persistence Interfaces ---
    
    def save_state(self, filepath: str):
        """Save narratives + FAISS index + index_map in one pickle."""
        import pickle, faiss
        print(f"\n--- 💾 Saving state to {filepath} ---")
        state = {
            "narratives": self.narratives,
            "index_bytes": faiss.serialize_index(self.index) if self.index and self.index.ntotal > 0 else None,
            "index_map": getattr(self, "index_map", []),
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("  ✅ State saved.")

    def load_state(self, filepath: str):
        """Load narratives + FAISS index + index_map from one pickle."""
        import os, pickle, faiss
        if not os.path.exists(filepath):
            print(f"\n--- 📂 No state file found at {filepath}. Starting fresh. ---")
            return

        print(f"\n--- 📂 Loading state from {filepath} ---")
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.narratives = state["narratives"]
        self.index_map  = state.get("index_map", [])
        idx_bytes = state.get("index_bytes")

        if idx_bytes is not None:
            self.index = faiss.deserialize_index(idx_bytes)
            print(f"  - ✅ Loaded FAISS index from pickle with {self.index.ntotal} statements.")
        else:
            # minimal fallback: empty index at correct dim
            import faiss
            self.index = faiss.IndexFlatIP(self.embedding_model.n_embd())
            print("  - ℹ️ No index in pickle; created empty index.")

        print(f"  ✅ State loaded. Found {len(self.narratives)} narratives.")

    def _on_narrative_file_ready(self, event):
        """
        Handles the NARRATIVE_FILE_READY event. This function now also:
        1. Writes a summary of all active narratives to a local file for debugging.
        2. Checks if the total number of narratives exceeds a hardcoded limit.
        3. Archives the lowest-scoring narratives to a designated folder if the limit is exceeded.
        4. Rebuilds the search index after any archival operation.
        """
        import os
        import json

        path = (event.payload or {}).get("filepath")
        if not path:
            return
        try:
            # Ingest the new narrative file first
            added = self.load_narrative_file(path)
            if added:
                print(f"✅ Ingested {added} new narratives from {os.path.basename(path)}")

            # --- MODIFICATION START ---

            # 1. Write a summary file for debugging purposes. This file is overwritten on each event.
            if self.narratives:
                summary_path = "narrative_summary.txt"
                print(f"  - Writing debug summary to {summary_path}...")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    # Create a header for the summary table
                    f.write(f"{'Narrative Title':<60} {'Score':<15} {'Hits':<10}\n")
                    f.write(f"{'-'*60} {'-'*15} {'-'*10}\n")
                    # Sort narratives by score for better readability in the summary
                    sorted_narratives = sorted(
                        self.narratives.values(),
                        key=lambda n: n.performance_score,
                        reverse=True
                    )
                    for n in sorted_narratives:
                        f.write(f"{n.narrative_title[:58]:<60} {n.performance_score:<15.4f} {n.hit_count:<10}\n")

            # 2. Archive old narratives if the total count exceeds the defined maximum.
            max_num_narratives = 100  # Hardcoded maximum number of active narratives
            if len(self.narratives) > max_num_narratives:
                num_to_archive = len(self.narratives) - max_num_narratives
                print(f"  ! Narrative count ({len(self.narratives)}) exceeds max ({max_num_narratives}). Archiving {num_to_archive}...")

                # Identify the narratives with the lowest scores to be archived
                narratives_to_archive = sorted(
                    self.narratives.values(),
                    key=lambda n: n.performance_score
                )[:num_to_archive]

                # Get the archive directory path from the config and ensure it exists
                archive_dir = self.config.ARCHIVED_NARRATIVE_PATH
                os.makedirs(archive_dir, exist_ok=True)

                archived_ids = []
                for n_to_archive in narratives_to_archive:
                    # Define the full path for the archived narrative's JSON file
                    archive_filepath = os.path.join(archive_dir, f"{n_to_archive.narrative_id}.json")
                    with open(archive_filepath, 'w', encoding='utf-8') as f_archive:
                        # Serialize the narrative object to a dictionary and save as JSON
                        json.dump(n_to_archive.to_dict(), f_archive, indent=2)

                    # Remove the archived narrative from the active dictionary
                    del self.narratives[n_to_archive.narrative_id]
                    archived_ids.append(n_to_archive.narrative_id)
                
                print(f"  - Archived and removed narratives: {', '.join(archived_ids)}")

                # 3. Rebuild the search index to reflect the removed narratives.
                print("  - Rebuilding search index after archival...")
                self._rebuild_index()

            # --- MODIFICATION END ---

        except Exception as e:
            print(f"  ! An error occurred during post-ingestion processing for {path}: {e}")

    def _load_narrative(self, filepath: str) -> List['Narrative']:
        """
        Loads, parses, and creates Narrative objects from a single file.

        Args:
            filepath: The full path to a single narrative .txt file.

        Returns:
            A list of Narrative objects found in the file.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Strip markdown code fences if they exist
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            
            raw_text = "".join(lines)
            # Clean trailing commas for more robust JSON parsing
            clean_text = re.sub(r',\s*(?=[}\]])', '', raw_text)
            data = json.loads(clean_text)

            # Use the Narrative.from_json factory for each item in the file
            narratives_in_file = [
                Narrative.from_json(narrative_data)
                for narrative_data in data.get('narratives', [])
            ]
            
            return narratives_in_file

        except Exception as e:
            print(f"  ! Error processing file '{os.path.basename(filepath)}': {e}")
            return []

    # --- Private Helper Methods ---

    def _apply_score_delta(self, event):
        nid = event.payload.get("narrative_id")
        d   = event.payload.get("delta", 0)
        n = self.narratives.get(nid)
        if n is None: return
        n.performance_score  = getattr(n, "performance_score", 0) + d
        print(f"  ↑ score[{nid}] = {n.performance_score}")

    def _initialize_models(self):
        from llama_cpp import Llama
        import faiss

        print(f"  - Loading embedding model from: {self.config.EMBEDDING_MODEL_PATH}")
        self.embedding_model = Llama(
            model_path=self.config.EMBEDDING_MODEL_PATH,
            embedding=True,
            n_gpu_layers=35,   # do CPU first to remove GPU VRAM variables
            n_ctx=512,
            n_parallel=64,   # <-- key fix: allow up to 64 sequences in parallel
            verbose=False
        )

        # Test immediately so you know if it's broken
        try:
            test_vec = self.embedding_model.embed(["hello world"])
            print(f"  - Test embedding OK. Dim = {len(test_vec[0])}")
        except Exception as e:
            raise RuntimeError(f"Embedding test failed: {e}")

        dim = len(test_vec[0])
        self.index = faiss.IndexFlatIP(dim)
        print("  - ✅ Real models initialized successfully.")

    # inside NarrativeManager

    def _chunk_input_text(self, text: str, window_size: int = 3, stride: int = 1):
        """Break large text into sliding windows of sentences."""
        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                raise RuntimeError("NLTK punkt not available")
        except Exception:
            import re
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        if not sentences:
            return []
        chunks = []
        for i in range(0, max(1, len(sentences) - window_size + 1), max(1, stride)):
            chunks.append(" ".join(sentences[i:i+window_size]))
        if not chunks:
            chunks = [" ".join(sentences)]
        return chunks

    def start_match_pipe(self, fifo_path: str = None, window_size: int = 3, stride: int = 1):
        """
        Read lines from named pipe, aggregates matches by ticker signal,
        and publishes one event per narrative with max similarity scores.
        """
        import threading
        import time
        from collections import defaultdict
        from datetime import datetime

        path = fifo_path or getattr(self.config, "TRANSCRIPT_PIPE_PATH", None)
        if not path:
            raise ValueError("TRANSCRIPT_PIPE_PATH not set and no fifo_path provided")

        def _loop():
            while True:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            text = line.strip()
                            if not text:
                                continue

                            # 1. Collect all raw match payloads from all chunks.
                            all_match_payloads = []
                            for chunk in self._chunk_input_text(text, window_size, stride):
                                payloads = self.match_text_stream(chunk)
                                all_match_payloads.extend(payloads)

                            if not all_match_payloads:
                                continue

                            # --- SIMPLIFIED AGGREGATION LOGIC START ---

                            # 2. Aggregate matches into a simple summary for each narrative.
                            aggregated_results = defaultdict(lambda: {
                                "affected_tickers": [],
                                "confirming_matches": 0,
                                "refuting_matches": 0,
                                "max_confirming_score": 0.0,
                                "max_refuting_score": 0.0
                            })

                            for payload in all_match_payloads:
                                nid = payload.get('narrative_id')
                                if not nid:
                                    continue

                                # Grab the static affected_tickers list, but only if we haven't already.
                                if not aggregated_results[nid]["affected_tickers"]:
                                    aggregated_results[nid]["affected_tickers"] = payload.get('affected_tickers', [])

                                # Get the data for the current match
                                event_type = payload.get('event_type')
                                score = payload.get('similarity_score', 0.0)

                                # Update counts and max scores based on the event_type
                                if event_type == 'CONFIRMING':
                                    aggregated_results[nid]["confirming_matches"] += 1
                                    current_max = aggregated_results[nid]["max_confirming_score"]
                                    aggregated_results[nid]["max_confirming_score"] = max(current_max, score)
                                elif event_type == 'REFUTING':
                                    aggregated_results[nid]["refuting_matches"] += 1
                                    current_max = aggregated_results[nid]["max_refuting_score"]
                                    aggregated_results[nid]["max_refuting_score"] = max(current_max, score)


                            # 3. Publish one new, simplified event per narrative.
                            for narrative_id, data in aggregated_results.items():
                                new_event_payload = {
                                    "narrative_id": narrative_id,
                                    "timestamp": datetime.now(),
                                    "affected_tickers": data["affected_tickers"],
                                    "confirming_matches": data["confirming_matches"],
                                    "refuting_matches": data["refuting_matches"],
                                    "max_confirming_score": float(data["max_confirming_score"]),
                                    "max_refuting_score": float(data["max_refuting_score"])
                                }

                                # Publish the new, consolidated event.
                                # Note: You may want to rename the event from NARRATIVE_MATCH to something
                                # like NARRATIVE_AGGREGATE to reflect its new purpose.
                                event = Event(name="NARRATIVE_MATCH", payload=new_event_payload)
                                event_bus.publish(event)
                                print(f"  📢 Published event for Narrative ID: {narrative_id}")
                                #print(f" Event:\n {event}")

                            # --- SIMPLIFIED AGGREGATION LOGIC END ---

                except FileNotFoundError:
                    print(f"  ! Pipe not found: {path}; retrying in 2s")
                    time.sleep(2)
                except Exception as e:
                    print(f"  ! Pipe error: {e}; reopening in 1s")
                    time.sleep(1)

        threading.Thread(target=_loop, daemon=True).start()
        print(f"▶️ Listening on pipe: {path} (window={window_size}, stride={stride})")

    def _rebuild_index(self):
        # manager._rebuild_index()
        import faiss, numpy as np
        print("  - Rebuilding search index (from cached embeddings)...")
        dim = self.embedding_model.n_embd()
        self.index = faiss.IndexFlatIP(dim)
        self.index_map = []

        all_vecs = []
        total_new = 0
        for n in self.narratives.values():
            total_new += n.compute_embeddings(self.embedding_model, force=False, suppress_output_ctx=suppress_output)
            for ev, s, i in n.iter_paraphrases():
                v = ev.get("paraphrased_embeddings", [None])[i]
                if v is not None:
                    all_vecs.append(v)
                    self.index_map.append({"narrative_id": n.narrative_id, "statement_text": s})

        if all_vecs:
            X = np.asarray(all_vecs, dtype=np.float32); faiss.normalize_L2(X); self.index.add(X)
        print(f"  - ✅ Index rebuilt with {self.index.ntotal} statements. (new embeds: {total_new})")

if __name__ == "__main__":
    
    # --- 1. Setup Environment and Config ---
    from config import main_config as cfg

    # This test now uses the REAL narrative directory and a dedicated test state file
    REAL_NARRATIVE_DIR = cfg.LLM_OUTPUT_DIR
    TEST_STATE_PATH = cfg.TEST_NARRATIVE_MANAGER_PATH

    # --- 2. Initialize Manager and Subscribers ---
    manager_config = {'SIMILARITY_THRESHOLD': 0.9}
    manager = NarrativeManager(cfg)
    
    # --- 3. Core Application Logic: Load or Initialize ---
    try:
        print("\n--- ▶️ Attempting to load manager state from disk... ---")
        manager.load_state(TEST_STATE_PATH)

        if not manager.narratives:
            print("\n--- ⚠️ No saved state found. Loading from the LIVE folder instead. ---")
            # --- CHANGE IS HERE: Point to the real directory ---
            manager.load_narratives_from_folder(REAL_NARRATIVE_DIR)
            # --------------------------------------------------

        # --- 4. Test Matching ---
        # This part of the test is now less predictable, as it depends on what's in the folder.
        # We'll just run a generic match and print the result.
        if manager.narratives:
            print(f"\n--- ▶️ Triggering a generic match event ---")
            manager.match_text_stream("Breaking news regarding inflation and the economy.")
            print("\n--- ✅ Match test complete. Check logs for published events. ---")
        else:
            print("\n--- ⚠️ No narratives were loaded, skipping match test. ---")

        # Finally, save the manager's state back to our designated test path
        manager.save_state(TEST_STATE_PATH)

    except:
        print("unknown error")