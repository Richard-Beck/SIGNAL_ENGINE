import os
import requests
import time
from datetime import datetime

import sys
sys.path.append('.') # Allows imports from the project root

from config import main_config as cfg

# --- Main Application ---

def process_payload(filepath, prompt_content):
    """
    Sends a single payload file to the API and saves the response.

    Args:
        filepath (str): The full path to the input payload file.
        prompt_content (str): The content of the prompt file.

    Returns:
        bool: True if processed successfully, False otherwise.
    """
    filename = os.path.basename(filepath)
    print(f"Processing file: {filename}...")

    try:
        # 1. Load payload from the specified file
        with open(filepath, "r", encoding="utf-8") as f:
            payload = f.read()

        # 2. Compose the full message for the API
        message = prompt_content + "\n\n" + payload

        # 3. Set up the API call headers and JSON data
        headers = {
            "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
            "HTTP-Referer": cfg.OPENROUTER_API_URL,
            "Content-Type": "application/json"
        }
        json_data = {
            "model": cfg.NARRATIVE_LLM_MODEL,
            "messages": [{"role": "user", "content": message}]
        }

        # 4. Send the request to the API
        response = requests.post(cfg.OPENROUTER_API_URL, headers=headers, json=json_data, timeout=180) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # 5. Extract the output content from the response
        output_content = response.json()["choices"][0]["message"]["content"]

        # 6. Save the output to a new file in the output directory
        # This prevents overwriting the same 'output.txt' file each time.
        output_filename = f"processed_{filename}"
        output_filepath = os.path.join(cfg.LLM_OUTPUT_DIR, output_filename)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(output_content)

        print(f"Successfully processed. Output saved to: {output_filepath}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error processing {filename}: API request failed - {e}")
    except (KeyError, IndexError) as e:
        print(f"Error processing {filename}: Could not parse API response - {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {filename}: {e}")

    return False

def run_generator_service():
    """
    Main continuous loop to monitor the directory and process files.
    This service is now robust against restarts.
    """
    # This set will be pre-populated with files that have already been
    # processed in previous sessions, making the service stateful.
    processed_files = set()

    # --- Robustness Logic: Pre-populate the processed_files set ---
    # On startup, sync state with the file system to avoid reprocessing.
    print("--- Payload Processor Started: Performing initial scan... ---")
    try:
        # Get all summary files that are candidates for processing.
        summary_filenames = {f for f in os.listdir(cfg.TRANSCRIPT_SUMMARY_DIR) if f.endswith('.txt')}

        # Get the filenames of summaries that have already been turned into narratives.
        # The output filename is "processed_" + original summary filename.
        narrative_filenames = {f.replace('processed_', '') for f in os.listdir(cfg.LLM_OUTPUT_DIR) if f.startswith('processed_')}

        # The intersection of these two sets gives us the files already processed.
        already_processed = summary_filenames.intersection(narrative_filenames)

        if already_processed:
            print(f"Found {len(already_processed)} summary files that have already been processed. They will be skipped.")
            processed_files.update(already_processed)
        else:
            print("No previously processed files found. Starting fresh.")

    except FileNotFoundError:
        print("Warning: Input or output directory not found during initial scan. Will create them as needed.")
    except Exception as e:
        print(f"An unexpected error occurred during initial scan: {e}")
    # ----------------------------------------------------------------

    # Load the prompt content once at the beginning.
    try:
        with open(cfg.NARRATIVE_GENERATION_PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt_content = f.read()
    except FileNotFoundError:
        print(f"Error: The prompt file was not found at '{cfg.NARRATIVE_GENERATION_PROMPT_FILE}'. Exiting.")
        return

    # Ensure the output directory exists.
    if not os.path.exists(cfg.LLM_OUTPUT_DIR):
        print(f"Output directory not found. Creating it at: {cfg.LLM_OUTPUT_DIR}")
        os.makedirs(cfg.LLM_OUTPUT_DIR)

    print(f"Watching for new files in: {os.path.abspath(cfg.TRANSCRIPT_SUMMARY_DIR)}")
    
    while True:
        try:
            # Get the current time to determine which file is "in-progress"
            now = datetime.now()
            current_hour_filename = f"output_{now.strftime('%Y-%m-%d_%H')}.txt"

            # Get a list of all .txt files in the input directory
            all_files = [f for f in os.listdir(cfg.TRANSCRIPT_SUMMARY_DIR) if f.endswith('.txt')]

            # Determine which files need to be processed
            files_to_process = []
            for filename in all_files:
                # Skip the file for the current hour as it's still being written to.
                if filename == current_hour_filename:
                    continue
                # Skip files that have already been successfully processed.
                if filename in processed_files:
                    continue
                files_to_process.append(filename)

            if not files_to_process:
                print(f"No new files to process. Waiting... (Last checked: {now.strftime('%H:%M:%S')})")
            else:
                print(f"\nFound {len(files_to_process)} new file(s) to process at {now.strftime('%H:%M:%S')}.")
                for filename in sorted(files_to_process): # Sort to process in chronological order
                    filepath = os.path.join(cfg.TRANSCRIPT_SUMMARY_DIR, filename)
                    if process_payload(filepath, prompt_content):
                        # If processing was successful, add the file to our set of processed files.
                        processed_files.add(filename)

            # Wait for the specified interval before checking again.
            time.sleep(cfg.FILE_POLLING_INTERVAL_SECONDS)

        except FileNotFoundError:
            print(f"Error: The input directory was not found at '{cfg.TRANSCRIPT_SUMMARY_DIR}'. Please check the path.")
            print(f"Retrying in {cfg.FILE_POLLING_INTERVAL_SECONDS} seconds...")
            time.sleep(cfg.FILE_POLLING_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n--- Payload Processor Shutting Down ---")
            break
        except Exception as e:
            print(f"A critical error occurred in the main loop: {e}")
            print(f"Retrying in {cfg.FILE_POLLING_INTERVAL_SECONDS} seconds...")
            time.sleep(cfg.FILE_POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_generator_service()


