#!/usr/bin/env python3
"""
Test veloce per verificare il nuovo sistema train/test/ground_truth
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def test_datasets_exist():
    """Verifica che tutti i dataset necessari esistano"""
    print("🔍 Testing dataset existence...")
    
    required_files = [
        "datasets/track1_live_events_train.csv",
        "datasets/track1_live_events_test.csv", 
        "datasets/track1_live_events_test_ground_truth.csv",
        "datasets/track2_documents_train.csv",
        "datasets/track2_documents_test.csv",
        "datasets/track2_documents_test_ground_truth.csv",
        "datasets/track3_music_train.csv",
        "datasets/track3_music_test.csv",
        "datasets/track3_music_test_ground_truth.csv",
        "datasets/track4_copyright_train.csv",
        "datasets/track4_copyright_test.csv",
        "datasets/track4_copyright_test_ground_truth.csv"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
        else:
            print(f"✅ {file}")
    
    if missing:
        print(f"\n❌ Missing files:")
        for file in missing:
            print(f"   - {file}")
        return False
    
    print("✅ All dataset files found!")
    return True

def test_ground_truth_format():
    """Verifica il formato dei file ground truth"""
    print("\n🔍 Testing ground truth format...")
    
    tests = [
        ("datasets/track1_live_events_test_ground_truth.csv", ["is_anomaly", "anomaly_type"]),
        ("datasets/track2_documents_test_ground_truth.csv", ["is_fraudulent", "fraud_type"]),
        ("datasets/track3_music_test_ground_truth.csv", ["is_anomaly", "anomaly_type"]),
        ("datasets/track4_copyright_test_ground_truth.csv", ["is_infringement", "infringement_type"])
    ]
    
    for file, expected_cols in tests:
        if Path(file).exists():
            df = pd.read_csv(file)
            found_cols = [col for col in expected_cols if col in df.columns]
            if found_cols:
                print(f"✅ {file}: found {found_cols}")
            else:
                print(f"❌ {file}: missing expected columns {expected_cols}")
                return False
        else:
            print(f"⚠️ {file}: file not found, skipping")
    
    return True

def test_evaluation_system():
    """Testa il sistema di valutazione"""
    print("\n🔍 Testing evaluation system...")
    
    try:
        from evaluate_submissions import SubmissionEvaluator
        
        evaluator = SubmissionEvaluator()
        print(f"✅ SubmissionEvaluator initialized")
        print(f"✅ Ground truths loaded for: {list(evaluator.ground_truths.keys())}")
        
        # Test con una submission di esempio
        example_submission = {
            "team_info": {
                "team_name": "Test Team",
                "members": ["Tester"],
                "track": "Track1",
                "submission_time": "2024-01-15T14:30:00Z"
            },
            "model_info": {
                "algorithm": "Test Algorithm",
                "features_used": ["feature1", "feature2"]
            },
            "results": {
                "predictions": [0, 1, 0, 1, 0] * 2000,  # Estendi per match ground truth size
                "scores": [-0.1, 0.8, -0.2, 0.7, -0.1] * 2000,
                "total_test_samples": 10000,
                "anomalies_detected": 5000
            },
            "metrics": {
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81,
                "auc_roc": 0.89
            }
        }
        
        # Test calcolo metriche reali
        real_metrics = evaluator.calculate_real_metrics(example_submission)
        if real_metrics:
            print(f"✅ Real metrics calculation works: {real_metrics}")
        else:
            print(f"⚠️ Real metrics calculation failed (might be expected if ground truth doesn't match)")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation system test failed: {e}")
        return False

def test_track_solutions():
    """Verifica che le soluzioni dei track siano state aggiornate correttamente"""
    print("\n🔍 Testing track solutions...")
    
    track_files = [
        "Track1_Solution/track1_anomaly_detection.py",
        "Track2_Solution/track2_document_fraud_detection.py", 
        "Track3_Solution/track3_music.py",
        "Track4_Solution/track4_copyright_infringement.py"
    ]
    
    for track_file in track_files:
        if Path(track_file).exists():
            with open(track_file, 'r') as f:
                content = f.read()
            
            # Verifica che contenga la nuova funzione load_train_test_datasets
            if "load_train_test_datasets" in content:
                print(f"✅ {track_file}: updated with new train/test loading")
            else:
                print(f"❌ {track_file}: missing load_train_test_datasets function")
                return False
        else:
            print(f"⚠️ {track_file}: file not found")
    
    return True

def test_github_actions():
    """Verifica che il workflow GitHub Actions sia presente"""
    print("\n🔍 Testing GitHub Actions workflow...")
    
    workflow_file = ".github/workflows/auto-leaderboard.yml"
    if Path(workflow_file).exists():
        print(f"✅ {workflow_file}: GitHub Actions workflow found")
        
        with open(workflow_file, 'r') as f:
            content = f.read()
            
        # Verifica elementi chiave
        checks = [
            ("submissions/submission_*.json", "trigger path"),
            ("evaluate_submissions.py", "evaluation script"),
            ("ground truth", "ground truth validation"),
            ("leaderboard.md", "leaderboard update")
        ]
        
        for check, desc in checks:
            if check in content:
                print(f"✅ {desc}: found in workflow")
            else:
                print(f"❌ {desc}: missing in workflow")
                return False
        
        return True
    else:
        print(f"❌ {workflow_file}: GitHub Actions workflow not found")
        return False

def main():
    """Esegue tutti i test"""
    print("🧪 SIAE Hackathon - New System Test")
    print("=" * 50)
    
    tests = [
        ("Dataset Files", test_datasets_exist),
        ("Ground Truth Format", test_ground_truth_format),
        ("Evaluation System", test_evaluation_system),
        ("Track Solutions", test_track_solutions),
        ("GitHub Actions", test_github_actions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System ready for hackathon!")
        print("\n💡 Next steps:")
        print("1. Participants can now use the updated track solutions")
        print("2. GitHub Actions will automatically evaluate submissions")
        print("3. Leaderboard will update automatically on each push")
    else:
        print("⚠️ Some tests failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("- Run: python generate_datasets.py")
        print("- Check all track solutions have been updated")
        print("- Verify GitHub Actions workflow is in place")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 