# src/trading/trading_manager.py
import os
import threading
import time
from datetime import datetime, time as dt_time

from src.events.event_bus import event_bus, Event
from src.trading.get_prices import download_price_data
from config import main_config as cfg

class EndOfDayScheduler:
    """
    A simple scheduler that runs a task once per day at a specific time.
    """
    def __init__(self, trigger_time_edt: dt_time, task: callable):
        self.trigger_time_edt = trigger_time_edt
        self.task = task
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_triggered_date = None

    def start(self):
        print(f"⏰ EndOfDayScheduler started. Will trigger at {self.trigger_time_edt} EDT.")
        self.thread.start()

    def _run(self):
        while True:
            now_edt = datetime.now(cfg.edt_tz)
            today = now_edt.date()

            # Check if it's past the trigger time and we haven't run for today yet
            if now_edt.time() >= self.trigger_time_edt and today != self.last_triggered_date:
                print(f"🚀 Triggering end-of-day task at {now_edt.strftime('%H:%M:%S')} EDT...")
                try:
                    self.task()
                    self.last_triggered_date = today
                    print("✅ End-of-day task completed successfully.")
                except Exception as e:
                    print(f"❌ Error during end-of-day task: {e}")
            
            # Sleep for a reasonable interval (e.g., 5 minutes) before checking again
            time.sleep(300)

class TradingManager:
    """
    Manages the end-of-day trading analysis pipeline.
    """
    def __init__(self, bus=event_bus):
        self.bus = bus
        self.scheduler = EndOfDayScheduler(
            trigger_time_edt=dt_time(16, 5), # Trigger at 4:05 PM EDT
            task=self.run_daily_analysis
        )
        self.scheduler.start()

    def run_daily_analysis(self):
        """
        The main task to be run at the end of the trading day.
        """
        print("\n--- 📈 Starting End-of-Day Trading Analysis ---")
        
        # 1. Download the latest price data for the day
        print("  - Step 1: Downloading daily price data...")
        price_file = download_price_data() # Modified to return the path
        if not price_file:
            print("  - ❌ Could not download price data. Aborting analysis.")
            return

        # 2. Publish an event to notify the system that analysis should begin
        # The NarrativeEvaluator will be listening for this.
        print("  - Step 2: Publishing event to trigger performance evaluation...")
        self.bus.publish(Event(
            "START_EOD_PERFORMANCE_ANALYSIS",
            {"price_file_path": price_file}
        ))
