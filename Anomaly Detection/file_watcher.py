#!/usr/bin/env python3
"""
SIAE Hackathon - File Watcher for Leaderboard Updates
Monitora la cartella submissions per aggiornamenti automatici
"""

import time
import os
import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SubmissionHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_update = time.time()
        self.cooldown = 5  # 5 seconds cooldown between updates
    
    def on_created(self, event):
        if not event.is_directory and "submission_" in event.src_path and event.src_path.endswith('.json'):
            self.update_leaderboard(f"New submission: {Path(event.src_path).name}")
    
    def on_modified(self, event):
        if not event.is_directory and "submission_" in event.src_path and event.src_path.endswith('.json'):
            self.update_leaderboard(f"Modified submission: {Path(event.src_path).name}")
    
    def update_leaderboard(self, reason):
        current_time = time.time()
        if current_time - self.last_update < self.cooldown:
            return  # Skip if too soon
        
        self.last_update = current_time
        
        print(f"\n🔄 {reason}")
        print(f"⏰ {time.strftime('%H:%M:%S')} - Updating leaderboard...")
        
        try:
            # Run evaluation
            result = subprocess.run([sys.executable, "evaluate_submissions.py"], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Leaderboard updated successfully!")
                
                # Try to auto-commit
                try:
                    subprocess.run(["git", "add", "leaderboard.md"], check=True)
                    subprocess.run(["git", "commit", "-m", f"🏆 Auto-update: {reason}", "--no-verify"], check=True)
                    print("📝 Changes committed automatically")
                except:
                    print("⚠️  Leaderboard updated but not committed (manual commit required)")
            else:
                print(f"❌ Error updating leaderboard: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("👀 SIAE Hackathon - File Watcher")
    print("Monitoring submissions/ for automatic leaderboard updates...")
    print("Press Ctrl+C to stop")
    
    submissions_path = Path("submissions")
    if not submissions_path.exists():
        print("❌ submissions/ directory not found")
        return
    
    event_handler = SubmissionHandler()
    observer = Observer()
    observer.schedule(event_handler, str(submissions_path), recursive=False)
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Stopping file watcher...")
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
