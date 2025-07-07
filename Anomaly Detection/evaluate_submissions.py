#!/usr/bin/env python3
"""
SIAE Hackathon - Automatic Submission Evaluation System
Processes submission files and updates the leaderboard automatically.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import glob
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class SubmissionEvaluator:
    def __init__(self, submissions_dir="submissions"):
        self.submissions_dir = Path(submissions_dir)
        self.track_truth_files = {
            "Track1": "Track1_Solution/live_events_with_anomalies.csv",
            "Track2": "Track2_Solution/documents_fraud_detection.csv",
            "Track3": "Track3_Solution/music_anomaly_detection_results.csv"
        }
        self.ground_truths = {}
        self.load_all_ground_truths()
        
    def load_all_ground_truths(self):
        """Load ground truth data for all tracks"""
        for track, truth_file in self.track_truth_files.items():
            try:
                truth_path = Path(truth_file)
                if truth_path.exists():
                    df = pd.read_csv(truth_path)
                    
                    if track == "Track1":
                        # For Track1: anomaly_type column
                        ground_truth = df['anomaly_type'].notna().astype(int).values
                        print(f"{track} ground truth loaded: {len(ground_truth)} events")
                        print(f"{track} true anomalies: {ground_truth.sum()}")
                    elif track == "Track2":
                        # For Track2: is_fraudulent column
                        ground_truth = df['is_fraudulent'].astype(int).values
                        print(f"{track} ground truth loaded: {len(ground_truth)} documents")
                        print(f"{track} true frauds: {ground_truth.sum()}")
                    elif track == "Track3":
                        # For Track3: is_anomaly column
                        ground_truth = df['is_anomaly'].astype(int).values
                        print(f"{track} ground truth loaded: {len(ground_truth)} tracks")
                        print(f"{track} true anomalies: {ground_truth.sum()}")
                    
                    self.ground_truths[track] = ground_truth
                else:
                    print(f"Warning: {track} ground truth file not found at {truth_path}")
                    # Create synthetic ground truth
                    if track == "Track1":
                        self.ground_truths[track] = np.random.choice([0, 1], size=10000, p=[0.9, 0.1])
                    elif track == "Track2":
                        self.ground_truths[track] = np.random.choice([0, 1], size=5000, p=[0.85, 0.15])
                    elif track == "Track3":
                        self.ground_truths[track] = np.random.choice([0, 1], size=25000, p=[0.92, 0.08])
                    print(f"Using synthetic ground truth for {track}")
                    
            except Exception as e:
                print(f"Error loading {track} ground truth: {e}")
                # Fallback synthetic data
                if track == "Track1":
                    self.ground_truths[track] = np.random.choice([0, 1], size=10000, p=[0.9, 0.1])
                elif track == "Track2":
                    self.ground_truths[track] = np.random.choice([0, 1], size=5000, p=[0.85, 0.15])
                elif track == "Track3":
                    self.ground_truths[track] = np.random.choice([0, 1], size=25000, p=[0.92, 0.08])
    
    def validate_submission(self, submission_data):
        """Validate submission format and content"""
        errors = []
        
        # Check required fields
        required_fields = ['team_info', 'model_info', 'results', 'metrics']
        for field in required_fields:
            if field not in submission_data:
                errors.append(f"Missing required field: {field}")
        
        # Check team_info
        if 'team_info' in submission_data:
            team_required = ['team_name', 'members', 'track']
            for field in team_required:
                if field not in submission_data['team_info']:
                    errors.append(f"Missing team_info field: {field}")
        
        # Check results format
        if 'results' in submission_data:
            results = submission_data['results']
            if 'predictions_sample' in results:
                predictions = results['predictions_sample']
                if not isinstance(predictions, list):
                    errors.append("predictions_sample should be a list")
                elif len(predictions) == 0:
                    errors.append("predictions_sample is empty")
                elif not all(p in [0, 1] for p in predictions):
                    errors.append("predictions_sample should contain only 0s and 1s")
        
        # Check metrics
        if 'metrics' in submission_data:
            metrics = submission_data['metrics']
            required_metrics = ['precision', 'recall', 'f1_score']
            for metric in required_metrics:
                if metric not in metrics:
                    errors.append(f"Missing metric: {metric}")
                elif not isinstance(metrics[metric], (int, float)):
                    errors.append(f"Metric {metric} should be numeric")
        
        return errors
    
    def calculate_innovation_score(self, submission_data):
        """Calculate innovation score based on features and techniques"""
        score = 0.0
        
        # Feature diversity (0-40 points)
        if 'features_used' in submission_data.get('model_info', {}):
            features = submission_data['model_info']['features_used']
            feature_score = min(len(features) * 2, 40)  # 2 points per feature, max 40
            score += feature_score
        
        # Algorithm complexity (0-30 points)
        if 'algorithm' in submission_data.get('model_info', {}):
            algorithm = submission_data['model_info']['algorithm'].lower()
            if 'ensemble' in algorithm or '+' in algorithm:
                score += 30  # Ensemble methods
            elif any(advanced in algorithm for advanced in ['neural', 'deep', 'autoencoder']):
                score += 25  # Deep learning
            elif any(ml in algorithm for ml in ['forest', 'svm', 'dbscan']):
                score += 20  # Advanced ML
            else:
                score += 10  # Basic methods
        
        # Feature engineering (0-30 points)
        if 'feature_engineering' in submission_data.get('model_info', {}):
            engineered = submission_data['model_info']['feature_engineering']
            score += min(len(engineered) * 5, 30)  # 5 points per engineered feature
        
        return min(score, 100) / 100  # Normalize to 0-1
    
    def calculate_business_score(self, submission_data):
        """Calculate business impact score"""
        score = 0.0
        
        # Performance efficiency (0-50 points)
        if 'performance_info' in submission_data:
            perf = submission_data['performance_info']
            
            # Training time (lower is better)
            if 'training_time_seconds' in perf:
                train_time = perf['training_time_seconds']
                if train_time < 10:
                    score += 20
                elif train_time < 60:
                    score += 15
                elif train_time < 300:
                    score += 10
                else:
                    score += 5
            
            # Memory usage (lower is better)
            if 'memory_usage_mb' in perf:
                memory = perf['memory_usage_mb']
                if memory < 100:
                    score += 15
                elif memory < 500:
                    score += 10
                elif memory < 1000:
                    score += 5
        
        # Interpretability (0-50 points)
        if 'anomaly_breakdown' in submission_data:
            breakdown = submission_data['anomaly_breakdown']
            if len(breakdown) > 0:
                score += 30  # Provides anomaly type breakdown
                if 'other' in breakdown and breakdown['other'] < sum(breakdown.values()) * 0.3:
                    score += 20  # Good classification of anomaly types
        
        return min(score, 100) / 100  # Normalize to 0-1
    
    def evaluate_submission(self, submission_file):
        """Evaluate a single submission file"""
        try:
            with open(submission_file, 'r') as f:
                submission_data = json.load(f)
            
            # Validate format
            errors = self.validate_submission(submission_data)
            if errors:
                return {
                    'team_name': submission_data.get('team_info', {}).get('team_name', 'Unknown'),
                    'valid': False,
                    'errors': errors,
                    'final_score': 0.0
                }
            
            # Extract metrics
            metrics = submission_data['metrics']
            
            # Calculate component scores
            technical_score = (
                metrics['f1_score'] * 0.25 +
                metrics.get('auc_roc', 0.5) * 0.15 +
                metrics['precision'] * 0.10
            )
            
            innovation_score = self.calculate_innovation_score(submission_data)
            business_score = self.calculate_business_score(submission_data)
            
            # Calculate final score
            final_score = technical_score * 0.5 + innovation_score * 0.3 + business_score * 0.2
            
            # Get submission info
            team_info = submission_data['team_info']
            track = team_info.get('track', 'Unknown')
            
            # Track-specific information
            track_specific_data = {}
            if track == "Track2" and 'track2_specific' in submission_data:
                track_specific_data = submission_data['track2_specific']
            elif track == "Track3" and 'track3_specific' in submission_data:
                track_specific_data = submission_data['track3_specific']
            
            return {
                'team_name': team_info['team_name'],
                'members': team_info['members'],
                'track': track,
                'submission_time': team_info.get('submission_time', 'Unknown'),
                'valid': True,
                'errors': [],
                'metrics': {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'auc_roc': metrics.get('auc_roc', 0.0)
                },
                'scores': {
                    'technical': technical_score,
                    'innovation': innovation_score,
                    'business': business_score,
                    'final': final_score
                },
                'algorithm': submission_data.get('model_info', {}).get('algorithm', 'Unknown'),
                'num_features': len(submission_data.get('model_info', {}).get('features_used', [])),
                'anomalies_detected': submission_data.get('results', {}).get('anomalies_detected', 0) or 
                                     submission_data.get('results', {}).get('frauds_detected', 0),
                'submission_file': submission_file.name,
                'track_specific': track_specific_data
            }
            
        except Exception as e:
            return {
                'team_name': 'Unknown',
                'valid': False,
                'errors': [f"Error processing file: {str(e)}"],
                'final_score': 0.0
            }
    
    def evaluate_all_submissions(self):
        """Evaluate all submissions in the directory"""
        submission_files = list(self.submissions_dir.glob("submission_*.json"))
        
        if not submission_files:
            print("No submission files found.")
            return []
        
        print(f"Found {len(submission_files)} submission files")
        
        results = []
        for file in submission_files:
            print(f"Evaluating {file.name}...")
            result = self.evaluate_submission(file)
            results.append(result)
        
        # Sort by final score (descending)
        results.sort(key=lambda x: x.get('scores', {}).get('final', 0), reverse=True)
        
        return results
    
    def generate_leaderboard(self, results):
        """Generate multi-track leaderboard markdown"""
        leaderboard_md = """# 🏆 SIAE Hackathon Leaderboard

*Ultimo aggiornamento: {timestamp}*

## 🌟 Overall Rankings

### 🥇 Top Teams Across All Tracks
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Overall top teams (best score from any track)
        team_best_scores = {}
        for result in results:
            if not result['valid']:
                continue
            team = result['team_name']
            track = result.get('track', 'Unknown')
            score = result['scores']['final']
            
            if team not in team_best_scores or score > team_best_scores[team]['score']:
                team_best_scores[team] = {
                    'score': score,
                    'track': track,
                    'result': result
                }
        
        # Sort by best score
        sorted_teams = sorted(team_best_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        leaderboard_md += "\n| Rank | Team | Best Score | Track | F1 | Algorithm | Members |\n"
        leaderboard_md += "|------|------|------------|-------|----|-----------|---------|\n"
        
        for i, (team_name, team_data) in enumerate(sorted_teams[:10]):
            rank = i + 1
            result = team_data['result']
            track = team_data['track']
            score = team_data['score']
            f1 = result['metrics']['f1_score']
            algorithm = result['algorithm']
            members = ", ".join(result['members'])
            
            leaderboard_md += f"| {rank} | {team_name} | {score:.3f} | {track} | {f1:.3f} | {algorithm} | {members} |\n"
        
        # Separate leaderboards by track
        tracks = set(r.get('track', 'Unknown') for r in results if r['valid'])
        
        for track in sorted(tracks):
            track_results = [r for r in results if r['valid'] and r.get('track') == track]
            
            if track == "Track1":
                track_title = "Track 1: Live Events Anomaly Detection"
                data_type = "Events"
            elif track == "Track2":
                track_title = "Track 2: Document Fraud Detection" 
                data_type = "Documents"
            elif track == "Track3":
                track_title = "Track 3: Music Anomaly Detection"
                data_type = "Tracks"
            else:
                track_title = f"{track}: Unknown Track"
                data_type = "Items"
            
            leaderboard_md += f"\n## {track_title}\n\n"
            leaderboard_md += "| Rank | Team | Score | F1 | Precision | Recall | AUC-ROC | Algorithm | Features | Members |\n"
            leaderboard_md += "|------|------|-------|----|-----------|---------|---------|-----------|------------|----------|\n"
        
            for i, result in enumerate(track_results):
                rank = i + 1
                team = result['team_name']
                score = result['scores']['final']
                metrics = result['metrics']
                algorithm = result['algorithm']
                num_features = result['num_features']
                members = ", ".join(result['members'])
                
                leaderboard_md += f"| {rank} | {team} | {score:.3f} | {metrics['f1_score']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['auc_roc']:.3f} | {algorithm} | {num_features} | {members} |\n"
        
        # Add invalid submissions
        invalid_submissions = [r for r in results if not r['valid']]
        if invalid_submissions:
            leaderboard_md += "\n## ❌ Invalid Submissions\n\n"
            for result in invalid_submissions:
                leaderboard_md += f"- **{result['team_name']}**: {', '.join(result['errors'])}\n"
        
        # Add detailed scores by track
        leaderboard_md += "\n## 📊 Detailed Scores by Track\n\n"
        
        for track in sorted(tracks):
            track_results = [r for r in results if r['valid'] and r.get('track') == track]
            if not track_results:
                continue
                
            if track == "Track1":
                track_title = "Track 1: Live Events Anomaly Detection"
            elif track == "Track2":
                track_title = "Track 2: Document Fraud Detection"
            elif track == "Track3":
                track_title = "Track 3: Music Anomaly Detection"
            else:
                track_title = f"{track}: Unknown Track"
                
            leaderboard_md += f"### {track_title}\n\n"
            
            for i, result in enumerate(track_results):
                rank = i + 1
                team = result['team_name']
                scores = result['scores']
                
                leaderboard_md += f"#### {rank}. {team}\n"
                leaderboard_md += f"- **Technical Score**: {scores['technical']:.3f} (50%)\n"
                leaderboard_md += f"- **Innovation Score**: {scores['innovation']:.3f} (30%)\n"
                leaderboard_md += f"- **Business Score**: {scores['business']:.3f} (20%)\n"
                leaderboard_md += f"- **Final Score**: {scores['final']:.3f}\n"
                
                if track == "Track1":
                    leaderboard_md += f"- **Anomalies Detected**: {result.get('anomalies_detected', 0)}\n\n"
                elif track == "Track2":
                    leaderboard_md += f"- **Frauds Detected**: {result.get('anomalies_detected', 0)}\n\n"
                elif track == "Track3":
                    leaderboard_md += f"- **Music Anomalies Detected**: {result.get('anomalies_detected', 0)}\n\n"
                else:
                    leaderboard_md += f"- **Issues Detected**: {result.get('anomalies_detected', 0)}\n\n"
        
        return leaderboard_md
    
    def save_leaderboard(self, leaderboard_md):
        """Save leaderboard to file"""
        with open("leaderboard.md", "w") as f:
            f.write(leaderboard_md)
        print("Leaderboard saved to leaderboard.md")
    
    def run_evaluation(self):
        """Run complete evaluation process"""
        print("=" * 50)
        print("SIAE Hackathon - Automatic Evaluation")
        print("=" * 50)
        
        # Evaluate all submissions
        results = self.evaluate_all_submissions()
        
        if not results:
            print("No valid submissions found.")
            return
        
        # Generate leaderboard
        leaderboard_md = self.generate_leaderboard(results)
        
        # Save leaderboard
        self.save_leaderboard(leaderboard_md)
        
        # Print top 3
        print("\n🏆 TOP 3 TEAMS:")
        for i, result in enumerate(results[:3]):
            if result['valid']:
                print(f"{i+1}. {result['team_name']} - Score: {result['scores']['final']:.3f}")
        
        print(f"\nTotal submissions processed: {len(results)}")
        print(f"Valid submissions: {len([r for r in results if r['valid']])}")
        print(f"Invalid submissions: {len([r for r in results if not r['valid']])}")

def main():
    """Main function"""
    evaluator = SubmissionEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
