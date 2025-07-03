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
    def __init__(self, submissions_dir="submissions", truth_file="Track1_Solution/live_events_with_anomalies.csv"):
        self.submissions_dir = Path(submissions_dir)
        self.truth_file = Path(truth_file)
        self.ground_truth = None
        self.load_ground_truth()
        
    def load_ground_truth(self):
        """Load ground truth data for evaluation"""
        try:
            if self.truth_file.exists():
                df = pd.read_csv(self.truth_file)
                # Create ground truth from anomaly_type column
                self.ground_truth = df['anomaly_type'].notna().astype(int).values
                print(f"Ground truth loaded: {len(self.ground_truth)} events")
                print(f"True anomalies: {self.ground_truth.sum()}")
            else:
                print(f"Warning: Ground truth file not found at {self.truth_file}")
                # Create synthetic ground truth for demonstration
                self.ground_truth = np.random.choice([0, 1], size=10000, p=[0.9, 0.1])
                print("Using synthetic ground truth for demonstration")
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            self.ground_truth = np.random.choice([0, 1], size=10000, p=[0.9, 0.1])
    
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
            
            return {
                'team_name': team_info['team_name'],
                'members': team_info['members'],
                'track': team_info.get('track', 'Unknown'),
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
                'anomalies_detected': submission_data.get('results', {}).get('anomalies_detected', 0),
                'submission_file': submission_file.name
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
        """Generate leaderboard markdown"""
        leaderboard_md = """# 🏆 SIAE Hackathon Leaderboard

*Ultimo aggiornamento: {timestamp}*

## Track 1: Live Events Anomaly Detection

| Rank | Team | Score | F1 | Precision | Recall | AUC-ROC | Algorithm | Features | Members |
|------|------|-------|----|-----------|---------|---------|-----------|------------|---------|
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for i, result in enumerate(results):
            if not result['valid']:
                continue
                
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
        
        # Add detailed scores
        leaderboard_md += "\n## 📊 Detailed Scores\n\n"
        for i, result in enumerate(results):
            if not result['valid']:
                continue
                
            rank = i + 1
            team = result['team_name']
            scores = result['scores']
            
            leaderboard_md += f"### {rank}. {team}\n"
            leaderboard_md += f"- **Technical Score**: {scores['technical']:.3f} (50%)\n"
            leaderboard_md += f"- **Innovation Score**: {scores['innovation']:.3f} (30%)\n"
            leaderboard_md += f"- **Business Score**: {scores['business']:.3f} (20%)\n"
            leaderboard_md += f"- **Final Score**: {scores['final']:.3f}\n"
            leaderboard_md += f"- **Anomalies Detected**: {result['anomalies_detected']}\n\n"
        
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
