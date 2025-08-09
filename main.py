# main.py
import os, sys, signal, threading, time
from config import main_config as cfg
from src.nlp.narrative_manager import NarrativeManager

# adjust import to your project layout if needed:
from src.data_ingestion.streamer import run_streamer_service   # or: from streamer import run_streamer_service

def ensure_fifo(path: str):
    if not os.path.exists(path):
        os.mkfifo(path)

def main():
    # 1) Ensure the transcript FIFO exists
    fifo = getattr(cfg, "TRANSCRIPT_PIPE_PATH", None)
    if not fifo:
        raise ValueError("TRANSCRIPT_PIPE_PATH is not set in config")
    ensure_fifo(fifo)

    # 2) Start the streamer service (daemon thread; it blocks internally)
    t_stream = threading.Thread(target=run_streamer_service, daemon=True)
    t_stream.start()
    print("▶️ Streamer service started")

    # 3) Start the narrative manager
    manager = NarrativeManager(cfg)

    # 4) Load prior state if present (restores FAISS; no re-embedding)
    state_path = getattr(cfg, "TEST_NARRATIVE_MANAGER_PATH", None)
    if state_path and os.path.exists(state_path):
        manager.load_state(state_path)

    # 5) Begin consuming the transcript pipe (uses cfg.TRANSCRIPT_PIPE_PATH)
    manager.start_match_pipe()  # chunking + match per window handled inside

    # 6) Graceful shutdown: save state on SIGINT/SIGTERM
    def _shutdown(*_):
        try:
            if state_path:
                manager.save_state(state_path)
        finally:
            print("\n👋 Shutdown complete")
            sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 7) Keep main thread alive
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
