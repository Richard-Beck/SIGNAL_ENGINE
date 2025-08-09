#!/usr/bin/env python3
import subprocess
import sys
import time
import tempfile
import os
import queue
from threading import Thread

class LiveTranscriber:
    """
    Handles capturing a live audio stream and transcribing it in chunks using Whisper.
    """
    def __init__(self, stream_url, chunk_duration=30, whisper_cli_path="~/tools/whisper.cpp/build/bin/whisper-cli", whisper_model_path="~/tools/whisper.cpp/models/ggml-base.en.bin"):
        """
        Initializes the LiveTranscriber.

        Args:
            stream_url (str): The URL of the live audio stream.
            chunk_duration (int): The duration of each audio chunk in seconds.
            whisper_cli_path (str): Path to the Whisper CLI executable.
            whisper_model_path (str): Path to the Whisper model file.
        """
        self.stream_url = stream_url
        self.chunk_duration = chunk_duration
        self.whisper_cli_path = os.path.expanduser(whisper_cli_path)
        self.whisper_model_path = os.path.expanduser(whisper_model_path)
        self._transcription_queue = queue.Queue()
        self._is_running = False

        # Validate paths on initialization
        if not os.path.exists(self.whisper_cli_path):
            raise FileNotFoundError(f"Whisper CLI not found at '{self.whisper_cli_path}'")
        if not os.path.exists(self.whisper_model_path):
            raise FileNotFoundError(f"Whisper model not found at '{self.whisper_model_path}'")

    def _capture_audio_chunks(self):
        """Internal method to run in a thread, capturing audio chunks."""
        print("🎙️  Starting audio capture...")
        while self._is_running:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_filename = tmp_file.name
                
                ffmpeg_command = [
                    'ffmpeg', '-loglevel', 'quiet', '-i', self.stream_url,
                    '-t', str(self.chunk_duration), '-ar', '16000', '-ac', '1',
                    '-c:a', 'pcm_s16le', '-y', temp_filename
                ]
                subprocess.run(ffmpeg_command, check=True)
                self._transcription_queue.put(temp_filename)
            except Exception as e:
                print(f"An error occurred in audio capture: {e}", file=sys.stderr)
                time.sleep(5) # Wait before retrying

    def transcribe_stream(self, output_dir=None):
        """
        A generator that captures, transcribes, and yields transcribed text chunks.

        Args:
            output_dir (str, optional): If provided, saves transcripts to this directory.

        Yields:
            str: A chunk of transcribed text.
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ Created transcript output directory: {output_dir}")

        self._is_running = True
        capture_thread = Thread(target=self._capture_audio_chunks, daemon=True)
        capture_thread.start()
        print("✍️  Starting transcription...")

        while self._is_running:
            try:
                audio_file_path = self._transcription_queue.get(timeout=1)
                
                try:
                    whisper_command = [
                        self.whisper_cli_path, '-m', self.whisper_model_path,
                        '-f', audio_file_path, '-otxt'
                    ]
                    subprocess.run(whisper_command, check=True, capture_output=True, text=True)
                    
                    transcript_path = audio_file_path + ".txt"
                    if os.path.exists(transcript_path):
                        with open(transcript_path, 'r') as f:
                            transcribed_text = f.read().strip()
                        
                        if transcribed_text:
                            yield transcribed_text
                            if output_dir:
                                out_path = os.path.join(output_dir, f"{int(time.time())}.txt")
                                with open(out_path, 'w') as f_out:
                                    f_out.write(transcribed_text)
                        
                        os.remove(transcript_path)

                except subprocess.CalledProcessError as e:
                    print(f"Error during transcription: {e.stderr}", file=sys.stderr)
                finally:
                    if os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
                    self._transcription_queue.task_done()
                    
            except queue.Empty:
                continue

    def stop(self):
        """Stops the audio capture and transcription process."""
        print("🛑 Stopping transcription service...")
        self._is_running = False
