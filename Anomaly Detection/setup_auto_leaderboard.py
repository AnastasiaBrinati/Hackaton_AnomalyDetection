#!/usr/bin/env python3
"""
SIAE Hackathon - Setup Automatic Leaderboard Updates
Configura il sistema per aggiornare automaticamente la leaderboard
"""

import os
import sys
import subprocess
from pathlib import Path
import stat

def create_git_hook():
    """Crea un Git hook per aggiornare la leaderboard automaticamente"""
    print("🔧 Creating Git post-commit hook...")
    
    # Path to git hooks directory
    git_hooks_dir = Path(".git/hooks")
    
    if not git_hooks_dir.exists():
        print("❌ Not in a Git repository or .git/hooks not found")
        return False
    
    # Create post-commit hook
    hook_content = '''#!/bin/bash
# SIAE Hackathon - Auto Leaderboard Update Hook

echo "🔄 Checking for new submissions..."

# Check if any submission files were changed
CHANGED_SUBMISSIONS=$(git diff --name-only HEAD~1 | grep "submissions/submission_.*\\.json" | wc -l)

if [ "$CHANGED_SUBMISSIONS" -gt 0 ]; then
    echo "📥 Found new submissions, updating leaderboard..."
    
    # Activate environment if exists
    if [ -d "hackathon_env" ]; then
        echo "🐍 Activating hackathon_env..."
        source hackathon_env/bin/activate
    fi
    
    # Update leaderboard
    python evaluate_submissions.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Leaderboard updated successfully!"
        
        # Auto-commit the updated leaderboard
        git add leaderboard.md
        git commit -m "🏆 Auto-update leaderboard" --no-verify
        
        echo "🎉 Leaderboard committed automatically"
    else
        echo "❌ Failed to update leaderboard"
    fi
else
    echo "ℹ️  No submission changes detected"
fi
'''
    
    hook_file = git_hooks_dir / "post-commit"
    
    try:
        with open(hook_file, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        st = os.stat(hook_file)
        os.chmod(hook_file, st.st_mode | stat.S_IEXEC)
        
        print(f"✅ Git hook created: {hook_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating Git hook: {e}")
        return False

def create_file_watcher():
    """Crea uno script di monitoraggio file per aggiornamenti in tempo reale"""
    print("👀 Creating file watcher script...")
    
    watcher_content = '''#!/usr/bin/env python3
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
        
        print(f"\\n🔄 {reason}")
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
        print("\\n👋 Stopping file watcher...")
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("file_watcher.py", 'w') as f:
            f.write(watcher_content)
        
        # Make executable
        os.chmod("file_watcher.py", 0o755)
        
        print("✅ File watcher created: file_watcher.py")
        return True
        
    except Exception as e:
        print(f"❌ Error creating file watcher: {e}")
        return False

def create_auto_update_script():
    """Crea script semplificato per aggiornamenti manuali"""
    print("🔄 Creating manual update script...")
    
    script_content = '''#!/bin/bash
# SIAE Hackathon - Manual Leaderboard Update

echo "🔄 Updating SIAE Hackathon Leaderboard..."

# Activate environment if exists
if [ -d "hackathon_env" ]; then
    echo "🐍 Activating hackathon_env..."
    source hackathon_env/bin/activate
fi

# Run evaluation
echo "📊 Running evaluation..."
python evaluate_submissions.py

if [ $? -eq 0 ]; then
    echo "✅ Leaderboard updated successfully!"
    
    # Show current top 3
    echo ""
    echo "🏆 CURRENT TOP 3:"
    head -20 leaderboard.md | tail -10
    
    # Ask if user wants to commit
    read -p "💾 Commit leaderboard changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add leaderboard.md
        git commit -m "🏆 Manual leaderboard update - $(date '+%H:%M:%S')"
        echo "📝 Leaderboard committed!"
    fi
else
    echo "❌ Failed to update leaderboard"
    exit 1
fi
'''
    
    try:
        with open("update_leaderboard_manual.sh", 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod("update_leaderboard_manual.sh", 0o755)
        
        print("✅ Manual update script created: update_leaderboard_manual.sh")
        return True
        
    except Exception as e:
        print(f"❌ Error creating manual update script: {e}")
        return False

def install_watchdog():
    """Installa la libreria watchdog se necessaria"""
    print("📦 Checking watchdog dependency...")
    
    try:
        import watchdog
        print("✅ watchdog already installed")
        return True
    except ImportError:
        print("📥 Installing watchdog...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "watchdog"], check=True)
            print("✅ watchdog installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install watchdog: {e}")
            print("💡 Install manually with: pip install watchdog")
            return False

def main():
    """Main setup function"""
    print("🚀 SIAE Hackathon - Auto Leaderboard Setup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("submissions").exists():
        print("❌ submissions/ directory not found. Run from project root.")
        return False
    
    success = True
    
    # 1. Create Git hook
    print("\\n1️⃣ Setting up Git hook...")
    if not create_git_hook():
        success = False
    
    # 2. Install dependencies
    print("\\n2️⃣ Installing dependencies...")
    if not install_watchdog():
        print("⚠️  File watcher will not work without watchdog")
    
    # 3. Create file watcher
    print("\\n3️⃣ Creating file watcher...")
    if not create_file_watcher():
        success = False
    
    # 4. Create manual update script
    print("\\n4️⃣ Creating manual update script...")
    if not create_auto_update_script():
        success = False
    
    # Summary
    print("\\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    if success:
        print("✅ Auto-leaderboard system configured!")
        print("\\n🎯 How to use:")
        print("   • Automatic (Git): Commits with submissions auto-update leaderboard")
        print("   • File monitoring: python file_watcher.py")
        print("   • Manual update: ./update_leaderboard_manual.sh")
        print("   • Direct evaluation: python evaluate_submissions.py")
        
        print("\\n💡 Recommendations:")
        print("   1. Test with: ./update_leaderboard_manual.sh")
        print("   2. For real-time monitoring: python file_watcher.py")
        print("   3. Git hook works automatically on commits")
        
        return True
    else:
        print("⚠️  Some components failed to setup")
        print("💡 Check error messages above and fix manually")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 