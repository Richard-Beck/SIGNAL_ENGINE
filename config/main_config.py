import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists.
# This is useful for keeping sensitive data like API keys out of version control.
load_dotenv()

# --- Path and Directory Settings ---
# A central place for all file and folder paths. Using os.path.join
# ensures that paths are created correctly for any operating system.

# Base directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This will be the signal_engine/ directory
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROMPTS_DIR = os.path.join(ROOT_DIR, "config", "prompts")

# Data folders
LLM_OUTPUT_DIR = os.path.join(DATA_DIR, "raw_narratives")
PROCESSED_NARRATIVES_DIR = os.path.join(DATA_DIR, "processed_outputs")
LOG_DIR = os.path.join(DATA_DIR, "logs")
TRANSCRIPT_SUMMARY_DIR = os.path.join(DATA_DIR, "transcript_summaries")

# Prompt files
NARRATIVE_GENERATION_PROMPT_FILE = os.path.join(PROMPTS_DIR, "make-narratives.txt")
NARRATIVE_MERGE_PROMPT_FILE = os.path.join(PROMPTS_DIR, "merge-narratives.txt")

# State and log files
LIVE_STATE_PREFIX = os.path.join(DATA_DIR, "live_db_state")
CANDIDATE_STATE_PREFIX = os.path.join(DATA_DIR, "candidate_db_state")
NARRATIVE_MANAGER_PATH = os.path.join(DATA_DIR, "narrative_manager.pkl")
TEST_NARRATIVE_MANAGER_PATH = os.path.join(DATA_DIR, "test_narrative_manager.pkl")
MATCH_LOG_FILE = os.path.join(LOG_DIR, "detected_matches.csv")
LIVE_RESULTS_FILE = os.path.join(LOG_DIR, "live_results.csv")
LLM_SERVER_LOG_FILE = os.path.join(LOG_DIR, "llm_server.log")

# Named pipe for real-time communication
TRANSCRIPT_PIPE_PATH = "/tmp/transcript_pipe2"


# --- API Configuration ---
# All settings related to external APIs.

# OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# The HTTP_REFERER is required by the OpenRouter API.
# You might need to change this if you deploy your application.
HTTP_REFERER = "http://localhost"


# --- Model Configuration ---
# Paths and settings for all local and remote models.

# LLM for Narrative Generation and Merging (via OpenRouter)
NARRATIVE_LLM_MODEL = "deepseek/deepseek-r1:free"

# Local Embedding Model (for narrative_matcher.py)
EMBEDDING_MODEL_PATH = os.path.expanduser("~/tools/llama.cpp/models/bge-large-en-v1.5-f16.gguf")

# Local LLM Server (for streamer.py transcript summarization)
LLM_SERVER_PATH = os.path.expanduser("~/tools/llama.cpp/build/bin/llama-server")
SUMMARIZATION_LLM_MODEL_PATH = os.path.expanduser("~/tools/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
LLM_GPU_LAYERS = 35

# Local Whisper Transcription Model
WHISPER_CLI_PATH = os.path.expanduser("~/tools/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL_PATH = os.path.expanduser("~/tools/whisper.cpp/models/ggml-base.en.bin")


# --- Service and Application Settings ---
# Parameters that control the behavior of the different scripts.

# streamer.py settings
STREAM_URL = "https://bloomberg-live-prod-us-east-1.s3.amazonaws.com/rad/Channel-RAD-AWS-virginia-1/Source-RadBOS-96-1_live.m3u8"
TRANSCRIPT_CHUNK_SIZE = 8 # Number of chunks to combine before sending to LLM
LLM_SERVER_HOST = "127.0.0.1"
LLM_SERVER_PORT = 8080
STREAMER_SYSTEM_PROMPT = """
You are a helpful assistant. Your task is to provide a concise summary of the following audio transcript. Focus on the key topics and main points discussed.
"""

# streamer.py - Hardening/Retry settings
MAX_CHUNK_RETRIES = 5
BACKOFF_BASE = 1.5
BACKOFF_MAX_SECONDS = 30
HEARTBEAT_INTERVAL_SECONDS = 15

# generate-triggers.py settings
FILE_POLLING_INTERVAL_SECONDS = 60

# realtime_detector.py and narrative_matcher.py settings
DEFAULT_SIMILARITY_THRESHOLD = 0.7
NARRATIVE_LIMIT = 40 # Max number of "live" narratives
MAX_FILES_PER_UPDATE = 20 # Max new files to process in one candidate update cycle
MAX_MAINTENANCE_CYCLES = 10 # Safety limit for the maintenance loop

# Scheduler timings for realtime_detector.py
CANDIDATE_MAINTENANCE_INTERVAL_SECONDS = 12 * 60 * 60  # 12 hours
PROMOTION_INTERVAL_SECONDS = 7 * 24 * 60 * 60      # 7 days
SCHEDULER_SLEEP_INTERVAL_SECONDS = 10 * 60         # 10 minutes