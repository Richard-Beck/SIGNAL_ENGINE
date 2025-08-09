import os
import time
import threading
import copy
from datetime import datetime
import sys
import multiprocessing # <-- Added for service management

# --- Setup Paths ---
sys.path.append('.') # Allows imports from the project root

# --- Imports from your project ---
from src.nlp.narrative_matcher import NarrativeMatcher
from config import main_config as cfg
# Import the functions that run our background services
from src.data_ingestion.streamer import run_streamer_service
from src.narrative_generation.generate_triggers import run_generator_service

# ==============================================================================
# All of your existing functions remain exactly the same.
# ==============================================================================

def run_candidate_maintenance(candidate_matcher: NarrativeMatcher):
    """
    Continuously runs the top-up, cluster, and merge cycle on the candidate database
    until it becomes stable.
    """
    print(f"[{datetime.now()}] --- Starting Candidate Database Maintenance ---")
    cycle_count = 0
    while cycle_count < cfg.MAX_MAINTENANCE_CYCLES:
        cycle_count += 1
        state_before_cycle = set(candidate_matcher.narratives.keys())
        
        if len(candidate_matcher.narratives) < cfg.NARRATIVE_LIMIT:
            candidate_matcher.update_from_folder(cfg.PROCESSED_NARRATIVES_DIR, cfg.MAX_FILES_PER_UPDATE)
        candidate_matcher.cluster_narratives()
        if candidate_matcher.narrative_clusters:
            for cluster in list(candidate_matcher.narrative_clusters):
                # NOTE: The original script used MERGE_PROMPT_FILE, which we updated in the config
                candidate_matcher.merge_cluster(cluster, cfg.NARRATIVE_MERGE_PROMPT_FILE, cfg.OPENROUTER_API_KEY)
        
        state_after_cycle = set(candidate_matcher.narratives.keys())
        if state_before_cycle == state_after_cycle:
            print("  Candidate database is stable.")
            break
    
    candidate_matcher.build_index()
    candidate_matcher.save_state(cfg.CANDIDATE_STATE_PREFIX)
    print(f"[{datetime.now()}] --- Candidate Maintenance Finished ---")

def promote_challengers(live_matcher: NarrativeMatcher, candidate_matcher: NarrativeMatcher):
    """
    Implements the 'Champion vs. Challenger' promotion logic.
    """
    print(f"[{datetime.now()}] --- Starting Promotion Cycle ---")
    
    live_matcher.apply_performance_scores(cfg.LIVE_RESULTS_FILE)
    
    combined_narratives = copy.deepcopy(live_matcher.narratives)
    combined_narratives.update(candidate_matcher.narratives)
    print(f"  Combined {len(live_matcher.narratives)} live and {len(candidate_matcher.narratives)} candidate narratives.")

    sorted_narratives = sorted(
        combined_narratives.values(),
        key=lambda n: n.get('performance_score', 0.0),
        reverse=True
    )
    
    new_live_narratives_list = sorted_narratives[:cfg.NARRATIVE_LIMIT]
    
    live_matcher.narratives = {n['narrative_id']: n for n in new_live_narratives_list}
    print(f"  Promotion complete. New live set has {len(live_matcher.narratives)} narratives.")
    
    live_matcher.build_index()
    live_matcher.save_state(cfg.LIVE_STATE_PREFIX)
    
    print(f"[{datetime.now()}] --- Promotion Finished ---")

def pipe_listener_thread(matcher: NarrativeMatcher):
    """Dedicated thread to listen to the named pipe and perform real-time matching."""
    if not os.path.exists(cfg.TRANSCRIPT_PIPE_PATH):
        os.mkfifo(cfg.TRANSCRIPT_PIPE_PATH)
    print("\n--- ✅ Real-Time Event Detection is LIVE (Listening against stable Live set) ---")
    while True:
        try:
            with open(cfg.TRANSCRIPT_PIPE_PATH, 'r') as pipe:
                full_text = pipe.read()
                if full_text:
                    print(f"[{datetime.now()}] Live event received. Matching...")
                    # CORRECTED: Using the variable name from our config file
                    detected_matches_df = matcher.match_chunk(full_text, threshold=cfg.DEFAULT_SIMILARITY_THRESHOLD)
                    if detected_matches_df.empty:
                        print("no matches")
                    if not detected_matches_df.empty:
                        print(f"--- 🗣️  LIVE MATCH! {len(detected_matches_df)} events found! ---")
                        print(detected_matches_df.to_string())
                        # CORRECTED: Fixed typo in variable name from ATCH_LOG_FILE
                        header = not os.path.exists(cfg.MATCH_LOG_FILE)
                        detected_matches_df.to_csv(cfg.MATCH_LOG_FILE, mode='a', header=header, index=False)
                        print(f"  Live matches logged to {cfg.MATCH_LOG_FILE}")
        except Exception as e:
            print(f"[Listener Thread] Error: {e}. Restarting...")
            time.sleep(5)

# ==============================================================================
# New main execution block that launches services and then runs your code.
# ==============================================================================

if __name__ == '__main__':
    print("--- 🚀 Launching Signal Engine Services ---")
    
    # Define the services to run as separate processes
    services = {
        "Streamer": run_streamer_service,
        "Narrative Generator": run_generator_service,
    }
    
    processes = []
    
    # Use a try...finally block to ensure cleanup happens
    try:
        # Start each background service in its own process
        for name, target_func in services.items():
            process = multiprocessing.Process(target=target_func, name=name)
            process.start()
            processes.append(process)
            print(f"  ✅ Service '{name}' started (PID: {process.pid})")
        
        # --- Now, run your original main logic in this primary process ---
        
        if not cfg.OPENROUTER_API_KEY:
            raise ValueError("API key not found. Please set OPENROUTER_API_KEY in your .env file.")

         # --- 1. Initial Setup ---
        print("## Initializing Main Detector: Loading Live and Candidate Databases ##")
        try:
            live_matcher = NarrativeMatcher.load_state(cfg.LIVE_STATE_PREFIX, cfg.EMBEDDING_MODEL_PATH)
            # --- ROBUSTNESS FIX ---
            # If the state was loaded but the index is missing, rebuild it now.
            if live_matcher.narratives and live_matcher.index is None:
                print("State loaded without an index. Rebuilding index now...")
                live_matcher.build_index()
                # Optionally, save the state again so the index exists for the next run
                live_matcher.save_state(cfg.LIVE_STATE_PREFIX)
                
        except FileNotFoundError:
            print("No saved 'Live' state found. Creating from scratch...")
            live_matcher = NarrativeMatcher(model_path=cfg.EMBEDDING_MODEL_PATH)
            live_matcher.initialize_from_folder(cfg.PROCESSED_NARRATIVES_DIR, narrative_limit=cfg.NARRATIVE_LIMIT)
            live_matcher.save_state(cfg.LIVE_STATE_PREFIX)

        # Apply the same logic for the candidate matcher
        try:
            candidate_matcher = NarrativeMatcher.load_state(cfg.CANDIDATE_STATE_PREFIX, cfg.EMBEDDING_MODEL_PATH)
            # --- ROBUSTNESS FIX ---
            if candidate_matcher.narratives and candidate_matcher.index is None:
                print("Candidate state loaded without an index. Rebuilding index now...")
                candidate_matcher.build_index()
                candidate_matcher.save_state(cfg.CANDIDATE_STATE_PREFIX)
                
        except FileNotFoundError:
            print("No saved 'Candidate' state found. Creating from live set...")
            candidate_matcher = NarrativeMatcher(model_path=cfg.EMBEDDING_MODEL_PATH)
            candidate_matcher.narratives = copy.deepcopy(live_matcher.narratives)
            candidate_matcher.last_processed_filename = live_matcher.last_processed_filename
            candidate_matcher.save_state(cfg.CANDIDATE_STATE_PREFIX)
        
        print("## System Initialized. ##")

        # 2. Start Real-Time Listener Thread
        listener = threading.Thread(target=pipe_listener_thread, args=(live_matcher,), daemon=True)
        listener.start()

        # 3. Start Main Scheduler Loop
        print(f"## Starting Main Scheduler Loop ##")
        last_maintenance_time = time.time()
        last_promotion_time = time.time()

        while True:
            if (time.time() - last_maintenance_time) >= cfg.CANDIDATE_MAINTENANCE_INTERVAL_SECONDS:
                print(f"\n## [{datetime.now()}] Waking up for scheduled CANDIDATE maintenance... ##")
                run_candidate_maintenance(candidate_matcher)
                last_maintenance_time = time.time()

            if (time.time() - last_promotion_time) >= cfg.PROMOTION_INTERVAL_SECONDS:
                print(f"\n## [{datetime.now()}] Waking up for scheduled weekly PROMOTION... ##")
                promote_challengers(live_matcher, candidate_matcher)
                print("  Refreshing candidate set from new live set...")
                candidate_matcher.narratives = copy.deepcopy(live_matcher.narratives) # Simplified the copy
                last_promotion_time = time.time()
                last_maintenance_time = time.time()

            time.sleep(cfg.SCHEDULER_SLEEP_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n--- 🛑 Shutdown signal received. Terminating all processes... ---")
        
    finally:
        # This will run when you press Ctrl+C
        for process in processes:
            if process.is_alive():
                print(f"  Terminating '{process.name}' (PID: {process.pid})...")
                process.terminate()
                process.join() # Wait for process to exit
        print("--- ✨ All services have been shut down. ---")