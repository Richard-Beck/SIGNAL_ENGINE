import os
import re
import json
import random
import pickle
import requests
import sys
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import faiss
from llama_cpp import Llama
import nltk
from typing import List, Dict, Union

# Import for clustering
from sklearn.cluster import DBSCAN

import sys
sys.path.append('.') # Allows imports from the project root

from config import main_config as cfg

# A more robust context manager to suppress C-level stdout/stderr.
@contextmanager
def suppress_output():
    """Temporarily suppresses stdout and stderr to silence unavoidable C-library warnings."""
    stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        devnull_fd = devnull.fileno()
        saved_stdout, saved_stderr = os.dup(stdout_fd), os.dup(stderr_fd)
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)

class NarrativeMatcher:
    """
    A class to load, process, synthesize, and match text against a database of narratives.
    """
    def __init__(self, model_path: str, default_similarity_threshold: float = 0.80, n_gpu_layers: int = -1):
        self.narratives = {}
        self.narrative_embeddings = None
        self.narrative_id_map = []
        self.narrative_clusters = []
        self.last_processed_filename = None
        
        self.index = None
        self.index_map = []
        self.default_similarity_threshold = default_similarity_threshold
        
        print(f"Loading embedding model from: {model_path}")
        try:
            with suppress_output():
                self.model = Llama(model_path=model_path, embedding=True, n_gpu_layers=n_gpu_layers, n_ctx=512, verbose=False)
            try: nltk.data.find('tokenizers/punkt')
            except nltk.downloader.DownloadError: print("Downloading NLTK sentence tokenizer data ('punkt')..."); nltk.download('punkt')
        except Exception as e: print(f"FATAL: Could not load embedding model. Error: {e}"); raise

    # --- High-Level Workflow Methods ---

    def initialize_from_folder(self, folder_path: str, narrative_limit: int = 40):
        """Initial creation workflow: Loads, clusters, and indexes narratives from scratch."""
        # This method is now corrected to pass the limit down to the loading function.
        self._load_narratives_incrementally(folder_path, narrative_limit)
        if self.narratives:
            self.cluster_narratives()
            self.build_index()

    def update_from_folder(self, folder_path: str, max_files_to_process: int = 40):
        """Incrementally loads new narratives from a folder since the last run."""
        print("\n--- Checking for new narratives to update database ---")
        initial_count = len(self.narratives)
        self._load_narratives_incrementally(folder_path, max_files_to_process)
        new_narratives_count = len(self.narratives) - initial_count
        if new_narratives_count > 0:
            print(f"  ✅ Ingested {new_narratives_count} new narratives. Database update complete.")
            print("  Remember to re-cluster, synthesize, and rebuild the index.")
        else:
            print("  No new narrative files found. Database is already up-to-date.")

    def synthesize_and_prune(self, merge_prompt_path: str, prune_prompt_path: str, final_narrative_limit: int = 10, api_key: str = None):
        """Runs the full synthesis and pruning pipeline."""
        print("\n--- Running Narrative Synthesis (Step 3) ---")
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print("Warning: API key not provided and OPENROUTER_API_KEY not set. Skipping LLM-based steps.")
                return

        self._synthesize_narratives(prompt_path=merge_prompt_path, api_key=api_key)
        
        if len(self.narratives) > final_narrative_limit:
            print("\n--- Running Narrative Pruning (Step 4) ---")
            self._prune_narratives(target_limit=final_narrative_limit, prompt_path=prune_prompt_path, api_key=api_key)

        print("\n--- Rebuilding index after synthesis and pruning ---")
        self.build_index()

    # --- Core Public Methods ---

    def cluster_narratives(self, similarity_threshold: float = 0.75, min_cluster_size: int = 2):
        """Embeds and clusters all current narratives based on their title and description."""
        print("\n--- Clustering Narratives ---")
        if len(self.narratives) < min_cluster_size:
            print("  Not enough narratives loaded to form clusters. Skipping.")
            return

        print(f"  Creating semantic embeddings for {len(self.narratives)} narratives...")
        self.narrative_id_map = list(self.narratives.keys())
        signatures_to_embed = [f"{self.narratives[nid].get('narrative_title', '')}. {self.narratives[nid].get('narrative_description', '')}" for nid in self.narrative_id_map]
        
        with suppress_output():
            narrative_embeddings_raw = np.array(self.model.embed(signatures_to_embed), dtype=np.float32)
            
        faiss.normalize_L2(narrative_embeddings_raw)
        self.narrative_embeddings = narrative_embeddings_raw
        
        print("  Clustering narratives with DBSCAN...")
        eps = np.sqrt(2 - (2 * similarity_threshold))
        db = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='euclidean').fit(self.narrative_embeddings)
        labels = db.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  ✅ Clustering complete. Found {n_clusters} clusters.")
        
        self.narrative_clusters = []
        for k in sorted(list(set(labels))):
            if k != -1:
                cluster_indices = np.where(labels == k)[0]
                self.narrative_clusters.append([self.narrative_id_map[i] for i in cluster_indices])

    # In the NarrativeMatcher class

    def merge_cluster(self, cluster_ids: List[str], prompt_path: str, api_key: str):
        """
        Merges a single cluster of narratives using an LLM, logs the transaction,
        and correctly updates the internal state.
        """
        if len(cluster_ids) < 2:
            print("  Cluster has less than 2 narratives, skipping merge.")
            return

        print(f"  Attempting to merge cluster: {cluster_ids}")

        # --- 1. Prepare Payload and Log Input ---
        cluster_narratives = [self.narratives[nid] for nid in cluster_ids if nid in self.narratives]
        
        # If the narratives were already deleted in a previous run, this list will be empty.
        if not cluster_narratives:
            print("  Narratives for this cluster no longer exist (likely already merged). Removing stale cluster.")
            # FIX: Remove the stale cluster to prevent re-processing.
            self.narrative_clusters.remove(cluster_ids)
            return

        payload = json.dumps({"narratives": cluster_narratives}, indent=2)

        # NEW: Log the input payload to a file.
        log_dir = "merges"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_log_path = os.path.join(log_dir, f"{timestamp}_input.json")
        with open(input_log_path, 'w', encoding='utf-8') as f:
            f.write(payload)
        print(f"  Input for merge saved to: {input_log_path}")

        # --- 2. Call LLM and Log Output ---
        response_text = self._call_llm(prompt_path, payload, api_key)
        
        if not response_text:
            print(f"  LLM returned no response for cluster. Skipping merge.")
            return
            
        # NEW: Log the output response to a file.
        output_log_path = os.path.join(log_dir, f"{timestamp}_output.json")
        with open(output_log_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        print(f"  Output from merge saved to: {output_log_path}")

        # --- 3. Update Narrative Database State ---
        # Delete the old narratives that were part of the merge.
        for nid in cluster_ids:
            if nid in self.narratives:
                del self.narratives[nid]

        # Add the new merged narrative(s) from the LLM response.
        try:
            new_data = json.loads(response_text)
            for new_narrative in new_data.get('narratives', []):
                # Use the ID from the LLM or create a new one.
                new_id = new_narrative.get("narrative_id", f"merged_{timestamp}")
                self.narratives[new_id] = new_narrative
                print(f"  ✅ Successfully merged cluster into new narrative: {new_id}")
        except json.JSONDecodeError:
            print(f"  ❌ Error: Could not decode LLM response into JSON for cluster.")
        
        # FIX: Remove the now-processed cluster from the list to prevent re-processing.
        if cluster_ids in self.narrative_clusters:
            self.narrative_clusters.remove(cluster_ids)
            
    def build_index(self):
        """Builds the FAISS index from the 'paraphrased_statements' of all current narratives."""
        if not self.narratives:
            print("Cannot build index: No narratives loaded.")
            return
        print("\n--- Building FAISS search index for paraphrased statements ---")
        all_statements, self.index_map = [], []
        for narrative_id, narrative_data in self.narratives.items():
            affected_tickers = narrative_data.get('affected_tickers', [])
            for event in narrative_data.get('key_events', []):
                event_id, event_type = event.get('event_id'), event.get('event_type')
                for statement in event.get('paraphrased_statements', []):
                    all_statements.append(statement)
                    self.index_map.append({"narrative_id": narrative_id, "event_id": event_id, "event_type": event_type, "affected_tickers": affected_tickers, "statement_text": statement})
        if not all_statements:
            print("Cannot build index: No 'paraphrased_statements' found.")
            return
        print(f"Creating embeddings for {len(all_statements)} statements...")
        with suppress_output():
            db_embeddings = np.array(self.model.embed(all_statements), dtype=np.float32)
        faiss.normalize_L2(db_embeddings)
        embedding_dim = db_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(db_embeddings)
        print(f"✅ FAISS index built successfully with {self.index.ntotal} vectors.")

    def match_chunk(self, text_chunk: str, threshold: float = None, debug: bool = False, debug_top_k: int = 3) -> Union[pd.DataFrame, Dict]:
        if not self.index:
            print("Error: Index not built.")
            return pd.DataFrame() if not debug else {}
        if not text_chunk or not text_chunk.strip():
            return pd.DataFrame() if not debug else {}
            
        similarity_threshold = threshold if threshold is not None else self.default_similarity_threshold

        # --- DEBUG MODE ---
        if debug:
            input_sub_chunks = self._chunk_input_text(text_chunk)
            if not input_sub_chunks: return {"summary_matches": [], "debug_details": []}
            
            with suppress_output():
                chunk_embeddings = np.array(self.model.embed(input_sub_chunks), dtype=np.float32)

            faiss.normalize_L2(chunk_embeddings)
            debug_details, summary_matches_map = [], {}

            for i, sub_chunk in enumerate(input_sub_chunks):
                embedding = chunk_embeddings[i:i+1]
                distances, indices = self.index.search(embedding, k=debug_top_k)
                top_k_results = []
                for j in range(debug_top_k):
                    if j >= len(indices[0]): continue
                    match_index, similarity_score = indices[0][j], distances[0][j]
                    match_info = self.index_map[match_index]
                    passes_threshold = similarity_score >= similarity_threshold
                    match_data = {"rank": j + 1, "similarity": float(similarity_score), "passes_threshold": passes_threshold, **match_info}
                    top_k_results.append(match_data)
                    if passes_threshold and (match_info['narrative_id'] not in summary_matches_map or similarity_score > summary_matches_map[match_info['narrative_id']]['similarity']):
                        summary_matches_map[match_info['narrative_id']] = match_data
                debug_details.append({"input_sub_chunk": sub_chunk, "top_k_results": top_k_results})
            
            summary_matches = sorted(summary_matches_map.values(), key=lambda x: x['similarity'], reverse=True)
            return {"summary_matches": summary_matches, "debug_details": debug_details}

        # --- DATAFRAME MODE (DEFAULT) ---
        else:
            input_sub_chunks = self._chunk_input_text(text_chunk)
            if not input_sub_chunks: return pd.DataFrame()
            
            with suppress_output():
                chunk_embeddings = np.array(self.model.embed(input_sub_chunks), dtype=np.float32)

            faiss.normalize_L2(chunk_embeddings)
            
            aggregated_matches = {}
            for embedding in chunk_embeddings:
                # Use range_search for efficiency: get all matches above the threshold.
                lims, D, I = self.index.range_search(embedding.reshape(1, -1), thresh=similarity_threshold)
                for i in range(lims[0], lims[1]):
                    match_index, similarity_score = I[i], D[i]
                    match_info = self.index_map[match_index]
                    
                    # Aggregate results by the event they belong to.
                    event_key = (match_info['narrative_id'], match_info['event_id'])
                    
                    if event_key not in aggregated_matches:
                        aggregated_matches[event_key] = {
                            'max_similarity': float(similarity_score),
                            'n_matches': 1,
                            'event_type': match_info['event_type'],
                            'affected_tickers': match_info['affected_tickers']
                        }
                    else:
                        agg = aggregated_matches[event_key]
                        agg['n_matches'] += 1
                        agg['max_similarity'] = max(agg['max_similarity'], float(similarity_score))
            
            if not aggregated_matches: return pd.DataFrame()

            # Format the aggregated results into a flat list for the DataFrame.
            records = []
            timestamp = datetime.now().isoformat()
            for (narrative_id, event_id), data in aggregated_matches.items():
                for ticker_info in data['affected_tickers']:
                    records.append({
                        "narrative_id": narrative_id,
                        "event": event_id,
                        "confirm/refute": data['event_type'],
                        "ticker": ticker_info['ticker'],
                        "action_if_narrative_true": ticker_info['position_if_true'],
                        "max_similarity": data['max_similarity'],
                        "n_matches": data['n_matches'],
                        "timestamp": timestamp
                    })
            
            return pd.DataFrame(records)


    # --- State Management ---

    # In your NarrativeMatcher class

    def save_state(self, file_prefix: str):
        """Saves the current state of the matcher to disk."""
        # CORRECTED: The file extensions are now appended directly to the prefix.
        state_data_path = f"{file_prefix}.pkl"
        index_path = f"{file_prefix}.faiss"
        
        print(f"\n--- Saving state to prefix: {file_prefix} ---")
        
        if self.index is None:
            print("Warning: Index is not built. Saving state without an index file.")
        else:
            faiss.write_index(self.index, index_path)
        
        state = {
            "narratives": self.narratives,
            "narrative_clusters": self.narrative_clusters,
            "index_map": self.index_map,
            "default_similarity_threshold": self.default_similarity_threshold,
            "last_processed_filename": self.last_processed_filename,
            "index_path": os.path.basename(index_path) if self.index else None
        }
        with open(state_data_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"✅ State successfully saved to {state_data_path}")

    @classmethod
    def load_state(cls, file_prefix: str, model_path: str, n_gpu_layers: int = -1) -> 'NarrativeMatcher':
        """Loads a saved state from disk and returns an initialized instance."""
        # CORRECTED: The file extensions are now appended directly to the prefix.
        state_data_path = f"{file_prefix}.pkl"
        index_path = f"{file_prefix}.faiss"

        if not os.path.exists(state_data_path):
            raise FileNotFoundError(f"State file '{state_data_path}' not found.")
        
        print(f"\n--- Loading state from prefix: {file_prefix} ---")
        instance = cls(model_path, n_gpu_layers=n_gpu_layers)
        
        with open(state_data_path, 'rb') as f:
            state = pickle.load(f)
        
        # ... (rest of the loading logic is the same)
        instance.narratives = state["narratives"]
        instance.narrative_clusters = state["narrative_clusters"]
        instance.index_map = state["index_map"]
        instance.default_similarity_threshold = state["default_similarity_threshold"]
        instance.last_processed_filename = state.get("last_processed_filename")
        
        if state.get("index_path") and os.path.exists(index_path):
            instance.index = faiss.read_index(index_path)
            print("  Index loaded from file.")
        else:
            print("  Warning: Index file not found or not specified in state. Index is not loaded.")
            
        print("✅ State successfully loaded.")
        return instance
# Add this method inside your NarrativeMatcher class

    def has_new_files(self, folder_path: str) -> bool:
        """Checks if there are new, unprocessed narrative files in the source folder."""
        if not os.path.isdir(folder_path):
            return False
        
        all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        
        # If we've never processed anything, any file is a new file.
        if not self.last_processed_filename:
            return len(all_files) > 0

        try:
            # Find the index of the last file we processed.
            last_index = all_files.index(self.last_processed_filename)
            # Return True if there are any files after it in the sorted list.
            return (last_index + 1) < len(all_files)
        except ValueError:
            # If the last file we processed has been deleted, it's safest to assume
            # there is new work to do by re-checking all files.
            print(f"Warning: Last processed file '{self.last_processed_filename}' not found. Assuming new files exist.")
            return True

    # In the NarrativeMatcher class
# This version now uses a weighted formula for the final performance score.

    def apply_performance_scores(self, results_filepath: str, accuracy_weight: float = 0.7, hit_weight: float = 0.3):
        """
        Loads a results file with hit_count and accuracy_score, then calculates
        and applies a final weighted performance_score to each narrative.
        """
        print("  Applying performance scores to live narratives...")

        # 1. Reset all performance keys to neutral defaults for all narratives
        for nid in self.narratives:
            self.narratives[nid]['performance_score'] = 0.0
            self.narratives[nid]['hit_count'] = 0
            self.narratives[nid]['accuracy_score'] = 0.0

        # 2. Exit if the results file doesn't exist
        if not os.path.exists(results_filepath):
            print(f"  Warning: Results file not found at '{results_filepath}'. All narratives will have a neutral score.")
            return

        try:
            # 3. Read the raw performance data
            results_df = pd.read_csv(results_filepath)
            
            # Find the maximum hit count for normalization
            max_hits = results_df['hit_count'].max() if not results_df.empty else 1
            if max_hits == 0: max_hits = 1 # Avoid division by zero

            # 4. Apply raw scores and calculate the final weighted score
            applied_count = 0
            for _, row in results_df.iterrows():
                nid = row['narrative_id']
                if nid in self.narratives:
                    # Store the raw metrics
                    accuracy = float(row['accuracy_score'])
                    hits = int(row['hit_count'])
                    self.narratives[nid]['accuracy_score'] = accuracy
                    self.narratives[nid]['hit_count'] = hits

                    # Normalize hit count to a 0-1 scale
                    normalized_hits = hits / max_hits

                    # Calculate the final weighted score
                    final_score = (accuracy_weight * accuracy) + (hit_weight * normalized_hits)
                    self.narratives[nid]['performance_score'] = final_score
                    applied_count += 1
            
            print(f"  Successfully applied and calculated performance scores for {applied_count} narratives.")

        except Exception as e:
            print(f"  Error reading or applying performance scores: {e}")    
    # --- Internal Helper Methods ---

    def _call_llm(self, prompt_path: str, payload: str, api_key: str, referer: str = "http://localhost") -> str:
        """Sends a request to the OpenRouter API and returns the text response."""
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f: prompt = f.read()

        message = f"{prompt}\n\n---\n\n{payload}"
        headers = {"Authorization": f"Bearer {api_key}", "HTTP-Referer": referer, "Content-Type": "application/json"}
        json_data = {"model": "deepseek/deepseek-r1:free", "messages": [{"role": "user", "content": message}]}
        
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=json_data)
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"]
            if output.strip().startswith("```json"):
                output = "\n".join(output.strip().split('\n')[1:-1])
            return output
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            return ""

    
    # In the NarrativeMatcher class

    def _load_narratives_incrementally(self, folder_path: str, max_files_to_process: int):
        """
        Loads narratives from a new batch of files, starting where it left off.
        """
        if not os.path.isdir(folder_path):
            print(f"Error: Folder not found at {folder_path}"); return
        
        all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        
        start_index = 0
        if self.last_processed_filename:
            try:
                start_index = all_files.index(self.last_processed_filename) + 1
            except ValueError:
                print(f"Warning: Last processed file '{self.last_processed_filename}' not found. Re-checking all files.")
        
        files_to_process = all_files[start_index : start_index + max_files_to_process]
        if not files_to_process:
            print("  No new narrative files found to process.")
            return

        print(f"  Ingesting up to {len(files_to_process)} new narrative file(s)...")
        for filename in files_to_process:
            filepath = os.path.join(folder_path, filename)
            try:
                # File parsing logic remains the same
                with open(filepath, 'r') as f: lines = f.readlines()
                if lines and lines[0].strip().startswith("```"): lines = lines[1:]
                if lines and lines[-1].strip() == "```": lines = lines[:-1]
                raw_text = "".join(lines)
                clean_text = re.sub(r',\s*(?=[}\]])', '', raw_text)
                data = json.loads(clean_text)
                for narrative in data.get('narratives', []):
                    if narrative_id := narrative.get('narrative_id'):
                        self.narratives[narrative_id] = narrative
                
                self.last_processed_filename = filename # Update after successfully processing
            except Exception as e:
                print(f"  Error processing file '{filename}': {e}")

    def _synthesize_narratives(self, prompt_path: str, api_key: str):
        """Iterates through clusters and merges them using an LLM."""
        for cluster_ids in list(self.narrative_clusters):
            self.merge_cluster(cluster_ids, prompt_path, api_key)

    def _prune_narratives(self, target_limit: int, prompt_path: str, api_key: str):
        """Reduces the number of narratives using pairwise A/B testing with an LLM."""
        while len(self.narratives) > target_limit:
            print(f"  Pruning... Current count: {len(self.narratives)}, Target: {target_limit}")
            ids_to_compare = random.sample(list(self.narratives.keys()), 2)
            narrative_a = self.narratives[ids_to_compare[0]]
            narrative_b = self.narratives[ids_to_compare[1]]
            payload = json.dumps([narrative_a, narrative_b], indent=2)
            response_text = self._call_llm(prompt_path, payload, api_key).strip()
            
            id_to_prune = response_text if response_text in ids_to_compare else random.choice(ids_to_compare)
            if id_to_prune in self.narratives:
                del self.narratives[id_to_prune]
                print(f"    - Pruned narrative '{id_to_prune}'.")

    def _chunk_input_text(self, text: str, window_size: int = 3, stride: int = 1) -> List[str]:
        """Helper to break down large text into sliding windows of sentences."""
        sentences = nltk.sent_tokenize(text)
        if not sentences: return []
        chunks = []
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunks.append(" ".join(sentences[i:i+window_size]))
        if not chunks and sentences: chunks = [" ".join(sentences)]
        return chunks