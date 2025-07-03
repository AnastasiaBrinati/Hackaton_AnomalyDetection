#!/usr/bin/env python3
"""
SIAE Hackathon - Automatic Leaderboard Update Script
Monitors git commits and updates the leaderboard when new submissions are detected.
"""

import subprocess
import os
import time
from pathlib import Path
from evaluate_submissions import SubmissionEvaluator

def check_for_new_submissions():
    """Check if there are new submission files in git"""
    try:
        # Get list of changed files in last commit
        result = subprocess.run(['git', 'diff', '--name-only', 'HEAD~1'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            changed_files = result.stdout.strip().split('\n')
            submission_files = [f for f in changed_files if 'submissions/submission_' in f and f.endswith('.json')]
            return submission_files
        return []
    except Exception as e:
        print(f"Error checking git changes: {e}")
        return []

def get_git_commit_info():
    """Get information about the latest commit"""
    try:
        # Get commit hash
        hash_result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                   capture_output=True, text=True)
        commit_hash = hash_result.stdout.strip()[:8] if hash_result.returncode == 0 else "unknown"
        
        # Get commit message
        msg_result = subprocess.run(['git', 'log', '-1', '--pretty=%s'], 
                                  capture_output=True, text=True)
        commit_msg = msg_result.stdout.strip() if msg_result.returncode == 0 else "No message"
        
        # Get commit author
        author_result = subprocess.run(['git', 'log', '-1', '--pretty=%an'], 
                                     capture_output=True, text=True)
        author = author_result.stdout.strip() if author_result.returncode == 0 else "Unknown"
        
        return {
            'hash': commit_hash,
            'message': commit_msg,
            'author': author
        }
    except Exception as e:
        print(f"Error getting git info: {e}")
        return {'hash': 'unknown', 'message': 'unknown', 'author': 'unknown'}

def send_notification(team_name, score, rank):
    """Send notification about new submission (could be extended to Slack, Discord, etc.)"""
    print(f"🔔 NEW SUBMISSION ALERT!")
    print(f"   Team: {team_name}")
    print(f"   Score: {score:.3f}")
    print(f"   Rank: #{rank}")
    print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

def main():
    """Main function for automatic leaderboard updates"""
    print("🚀 SIAE Hackathon - Automatic Leaderboard Monitor")
    print("=" * 60)
    
    # Check if we're in a git repository
    if not Path('.git').exists():
        print("❌ Not in a git repository. Please run from project root.")
        return
    
    # Check for new submissions
    new_submissions = check_for_new_submissions()
    
    if new_submissions:
        print(f"📥 Found {len(new_submissions)} new submission(s):")
        for sub in new_submissions:
            print(f"   - {sub}")
        
        # Get commit info
        commit_info = get_git_commit_info()
        print(f"\n📋 Commit Info:")
        print(f"   Hash: {commit_info['hash']}")
        print(f"   Author: {commit_info['author']}")
        print(f"   Message: {commit_info['message']}")
        
        # Update leaderboard
        print(f"\n🔄 Updating leaderboard...")
        evaluator = SubmissionEvaluator()
        evaluator.run_evaluation()
        
        # Parse results to send notifications
        try:
            results = evaluator.evaluate_all_submissions()
            valid_results = [r for r in results if r['valid']]
            
            if valid_results:
                # Send notifications for top performers or recent submissions
                for i, result in enumerate(valid_results[:3]):  # Top 3
                    send_notification(result['team_name'], 
                                    result['scores']['final'], 
                                    i + 1)
        except Exception as e:
            print(f"Error sending notifications: {e}")
        
        print(f"\n✅ Leaderboard updated successfully!")
        
    else:
        print("ℹ️  No new submissions detected in latest commit.")
        print("💡 To test the system, add a submission file to submissions/")

def monitor_mode():
    """Continuous monitoring mode (for development/testing)"""
    print("🔍 Starting continuous monitoring mode...")
    print("   Press Ctrl+C to stop")
    
    last_check = time.time()
    
    try:
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            # Simple check based on file modification time
            submissions_dir = Path("submissions")
            if submissions_dir.exists():
                latest_mod = 0
                for file in submissions_dir.glob("submission_*.json"):
                    mod_time = file.stat().st_mtime
                    if mod_time > latest_mod:
                        latest_mod = mod_time
                
                if latest_mod > last_check:
                    print(f"\n🔄 Detected new/modified submission files...")
                    evaluator = SubmissionEvaluator()
                    evaluator.run_evaluation()
                    last_check = time.time()
                    print(f"✅ Leaderboard updated at {time.strftime('%H:%M:%S')}")
    
    except KeyboardInterrupt:
        print(f"\n👋 Monitoring stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        monitor_mode()
    else:
        main() 