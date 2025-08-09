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
        similarity threshold and publishes a 'NARRATIVE_MATCH' event for each one.
        This version uses real embeddings and FAISS range_search.

        Args:
            text: The input text from the real-time stream.

        Returns:
            A list of all match_info dictionaries, or an empty list if no matches were found.
        """
        import faiss
        import numpy as np

        if self.index.ntotal == 0:
            print("  ! Index is empty. No matching possible.")
            return []

        # 1. Embed the input text once, silencing the C-library output.
        with suppress_output():
            query_embedding = np.array(self.embedding_model.embed([text]), dtype='float32')
        
        # 2. CRITICAL: The query vector must also be normalized.
        faiss.normalize_L2(query_embedding)

        # 3. Use `range_search` to find all vectors with a similarity score
        #    (dot product) greater than or equal to the threshold.
        threshold = self.config.DEFAULT_SIMILARITY_THRESHOLD
        lims, distances, indices = self.index.range_search(query_embedding, thresh=threshold)

        found_matches = []
        
        # 4. Iterate through the returned results from range_search.
        for i in range(len(indices)):
            similarity_score = distances[i]
            match_index = indices[i]
            
            # --- Match Found: Start the event workflow ---
            
            match_info = self.index_map[match_index]
            matched_narrative_id = match_info['narrative_id']
            matched_narrative = self.narratives[matched_narrative_id]

            # 5. Call record_match to update state and get the event payload.
            event_payload = matched_narrative.record_match(
                match_timestamp=datetime.now(),
                matched_text=text,
                similarity_score=similarity_score
            )

            # 6. Wrap the payload in a generic Event and publish.
            event_to_publish = Event(name="NARRATIVE_MATCH", payload=event_payload)
            event_bus.publish(event_to_publish)

            # 7. Add the successful match to our return list.
            found_matches.append(match_info)

        if found_matches:
            print(f"  ✅ Found and published {len(found_matches)} matches.")

        return found_matches

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
        n.score = getattr(n, "score", 0) + d
        print(f"  ↑ score[{nid}] = {n.score}")

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
        """Read lines from named pipe; chunk & match each window."""
        import threading, time

        path = fifo_path or getattr(self.config, "TRANSCRIPT_PIPE_PATH", None)
        if not path:
            raise ValueError("TRANSCRIPT_PIPE_PATH not set and no fifo_path provided")

        def _loop():
            while True:
                try:
                    # blocks until writer connects
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            text = line.strip()
                            if not text:
                                continue
                            for chunk in self._chunk_input_text(text, window_size, stride):
                                self.match_text_stream(chunk)
                except FileNotFoundError:
                    print(f"  ! Pipe not found: {path}; retrying in 2s")
                    time.sleep(2)
                except Exception as e:
                    print(f"  ! Pipe error: {e}; reopening in 1s")
                    time.sleep(1)

        threading.Thread(target=_loop, daemon=True).start()
        print(f"▶️ Listening on pipe: {path} (window={window_size}, stride={stride})")

    def _rebuild_index(self):
        """
        Rebuild FAISS index using sequential embeddings with a simple progress bar.
        """
        import faiss, numpy as np

        print("  - Rebuilding search index with real embeddings...")
        dim = self.embedding_model.n_embd()
        self.index = faiss.IndexFlatIP(dim)
        self.index_map = []

        # collect paraphrases
        statements = []
        for n in self.narratives.values():
            for ev in n.key_events:
                for s in ev.get('paraphrased_statements', []):
                    if s and s.strip():
                        statements.append(s.strip())
                        self.index_map.append({"narrative_id": n.narrative_id, "statement_text": s.strip()})

        if not statements:
            print("  - ✅ Index rebuilt with 0 statements.")
            return

        # embed sequentially with progress bar
        vecs, dropped = [], 0
        for i, s in enumerate(tqdm(statements, desc="Embedding statements", unit="stmt")):
            try:
                with suppress_output():
                    v = self.embedding_model.embed([s])
                vecs.append(v[0])
            except Exception:
                dropped += 1
                self.index_map[i] = None

        # prune dropped entries
        if dropped:
            self.index_map = [m for m in self.index_map if m is not None]
            print(f"  ! Dropped {dropped} statements due to embed errors.")

        if not vecs:
            print("  - ✅ Index rebuilt with 0 statements.")
            return

        X = np.asarray(vecs, dtype=np.float32)
        faiss.normalize_L2(X)
        self.index.add(X)
        print(f"  - ✅ Index rebuilt with {self.index.ntotal} statements.")



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