#!/bin/bash
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
