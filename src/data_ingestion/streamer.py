#!/usr/bin/env python3
import subprocess
import sys
import time
import os
import requests
from datetime import datetime
from src.data_ingestion.transcriber import LiveTranscriber
import errno

import sys
sys.path.append('.') # Allows imports from the project root

from config import main_config as cfg

# --- Core Functions ---

def start_llm_server():
    """Starts the llama-server as a background process."""
    print(f"🚀 Starting LLM server. Logging to '{cfg.LLM_SERVER_LOG_FILE}'")
    command = [
        cfg.LLM_SERVER_PATH, "-m", cfg.SUMMARIZATION_LLM_MODEL_PATH, "--host", cfg.LLM_SERVER_HOST,
        "--port", str(cfg.LLM_SERVER_PORT), "-ngl", str(cfg.LLM_GPU_LAYERS)
    ]
    with open(cfg.LLM_SERVER_LOG_FILE, 'w') as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=log_file)
    time.sleep(10)
    return process


def get_hourly_output_filepath():
    """Determines the output file path based on the current hour."""
    now = datetime.now()
    filename = f"output_{now.strftime('%Y-%m-%d_%H')}.txt"
    return os.path.join(cfg.TRANSCRIPT_SUMMARY_DIR, filename)


def analyze_and_save(transcript_content):
    """Sends content to LLM and appends the response to the hourly file."""
    server_url = f"http://{cfg.LLM_SERVER_HOST}:{cfg.LLM_SERVER_PORT}/completion"
    full_prompt = f"<s>[INST] {cfg.STREAMER_SYSTEM_PROMPT}\n\n--- TRANSCRIPT ---\n{transcript_content}\n\n[/INST]"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": full_prompt, "n_predict": 1024, "temperature": 0.2}

    try:
        response = requests.post(server_url, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        llm_output = response.json().get('content', '').strip()

        if llm_output:
            output_filepath = get_hourly_output_filepath()
            separator = f"\n\n--- Analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n"
            with open(output_filepath, 'a') as f:
                f.write(separator + llm_output)
                f.flush(); os.fsync(f.fileno())
            print(f"✅ Appended analysis to {os.path.basename(output_filepath)}")
        else:
            print("🧠 LLM returned an empty response. Nothing to save.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error communicating with LLM server: {e}", file=sys.stderr)


def write_to_pipe_safe(pipe_path, text):
    try:
        fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
        with os.fdopen(fd, 'w') as pipe:
            pipe.write(text)
    except OSError as e:
        if e.errno == errno.ENXIO:
            # No reader is connected to the pipe
            print("⚠️ No consumer connected to named pipe. Skipping write.")
        else:
            raise


def run_streamer_service():
    """Main function to run the continuous analysis pipeline."""
    for path in [cfg.WHISPER_CLI_PATH, cfg.WHISPER_MODEL_PATH, cfg.LLM_SERVER_PATH, cfg.SUMMARIZATION_LLM_MODEL_PATH]:
        if not os.path.exists(path):
            print(f"❌ CRITICAL: Required file not found at '{path}'", file=sys.stderr)
            sys.exit(1)
            
    if not os.path.exists(cfg.TRANSCRIPT_SUMMARY_DIR):
        os.makedirs(cfg.TRANSCRIPT_SUMMARY_DIR)
        print(f"✅ Created output directory: {cfg.TRANSCRIPT_SUMMARY_DIR}")

    llm_process = None
    try:
        llm_process = start_llm_server()
        if llm_process.poll() is not None:
            raise RuntimeError(f"LLM Server failed to start. Check '{cfg.LLM_SERVER_LOG_FILE}'.")

        # Correctly instantiate the imported LiveTranscriber class
        transcriber = LiveTranscriber(
            stream_url=cfg.STREAM_URL,
            whisper_cli_path=cfg.WHISPER_CLI_PATH,
            whisper_model_path=cfg.WHISPER_MODEL_PATH
        )
        transcript_generator = transcriber.transcribe_stream()
        
        print(f"🟢 System is running. Combining {cfg.TRANSCRIPT_CHUNK_SIZE} chunks per analysis.")
       
        if not os.path.exists(cfg.TRANSCRIPT_PIPE_PATH):
            os.mkfifo(cfg.TRANSCRIPT_PIPE_PATH)
            print(f"✅ Created named pipe at: {cfg.TRANSCRIPT_PIPE_PATH}")
        while True:
            chunk_buffer = []
            print(f"Collecting {cfg.TRANSCRIPT_CHUNK_SIZE} transcript chunks...")

            i = 0
            last_heartbeat = time.time()
            while i < cfg.TRANSCRIPT_CHUNK_SIZE:
                # Heartbeat so "quiet" isn't mistaken for hang
                if time.time() - last_heartbeat >= cfg.HEARTBEAT_INTERVAL_SECONDS:
                    print(f"[{datetime.now()}] Alive. progress={i}/{cfg.TRANSCRIPT_CHUNK_SIZE}", flush=True)
                    last_heartbeat = time.time()

                # Per-chunk retry with exponential backoff
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        chunk = next(transcript_generator)
                        break
                    except StopIteration:
                        # regenerate generator cleanly
                        transcript_generator = transcriber.transcribe_stream()
                        continue
                    except subprocess.CalledProcessError as e:
                        backoff = min(cfg.BACKOFF_BASE ** attempt, cfg.BACKOFF_MAX)
                        print(f"[{datetime.now()}] ffmpeg failed (rc={e.returncode}). Retry {attempt}/{cfg.MAX_CHUNK_RETRIES} in {backoff:.1f}s…", flush=True)
                        if attempt >= cfg.MAX_CHUNK_RETRIES:
                            raise
                        time.sleep(backoff)
                        continue
                    except Exception as e:
                        backoff = min(cfg.BACKOFF_BASE ** attempt, cfg.BACKOFF_MAX)
                        print(f"[{datetime.now()}] chunk error: {e!r}. Retry {attempt}/{cfg.MAX_CHUNK_RETRIES} in {backoff:.1f}s…", flush=True)
                        if attempt >= cfg.MAX_CHUNK_RETRIES:
                            raise
                        time.sleep(backoff)
                        # Recreate generator in case it's poisoned
                        transcript_generator = transcriber.transcribe_stream()
                        continue

                write_to_pipe_safe(cfg.TRANSCRIPT_PIPE_PATH, chunk + "\n\n")
                chunk_buffer.append(chunk)
                i += 1
                print(f"  Got chunk {i}/{cfg.TRANSCRIPT_CHUNK_SIZE}")

            combined_transcript = "\n\n".join(chunk_buffer)
            print("🧠 Chunks collected. Sending for analysis...")
            analyze_and_save(combined_transcript)
            print("-" * 50)

    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"\n❌ Fatal error in main loop: {e}", file=sys.stderr)
    finally:
        if llm_process and llm_process.poll() is None:
            print("Terminating LLM server...")
            llm_process.terminate()
            llm_process.wait()
            print("LLM Server terminated.")

if __name__ == "__main__":
    run_streamer_service()

