#!/usr/bin/env python3
"""
SIAE Hackathon - Test Evaluation System
Script per testare il sistema di valutazione automatica
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def test_submissions_directory():
    """Test che la cartella submissions esista e contenga i file necessari"""
    print("🧪 Testing submissions directory...")
    
    submissions_dir = Path("submissions")
    if not submissions_dir.exists():
        print("❌ FAIL: submissions/ directory not found")
        return False
    
    example_file = submissions_dir / "submission_example.json"
    if not example_file.exists():
        print("❌ FAIL: submission_example.json not found")
        return False
    
    # Test formato JSON del file di esempio
    try:
        with open(example_file, 'r') as f:
            example_data = json.load(f)
        
        required_fields = ['team_info', 'model_info', 'results', 'metrics']
        for field in required_fields:
            if field not in example_data:
                print(f"❌ FAIL: {field} missing in example submission")
                return False
        
        print("✅ PASS: submissions directory and example file OK")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ FAIL: Invalid JSON in example file: {e}")
        return False

def test_evaluation_script():
    """Test che lo script di valutazione funzioni"""
    print("🧪 Testing evaluation script...")
    
    if not Path("evaluate_submissions.py").exists():
        print("❌ FAIL: evaluate_submissions.py not found")
        return False
    
    try:
        # Test import
        result = subprocess.run([sys.executable, "-c", "from evaluate_submissions import SubmissionEvaluator; print('Import OK')"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ PASS: evaluate_submissions.py imports successfully")
            return True
        else:
            print(f"❌ FAIL: Import error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FAIL: Evaluation script timeout")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error testing evaluation script: {e}")
        return False

def test_track1_solution():
    """Test che la soluzione Track1 esista e abbia la funzione generate_submission"""
    print("🧪 Testing Track1 solution...")
    
    solution_file = Path("Track1_Solution/track1_anomaly_detection.py")
    if not solution_file.exists():
        print("❌ FAIL: Track1_Solution/track1_anomaly_detection.py not found")
        return False
    
    try:
        with open(solution_file, 'r') as f:
            content = f.read()
        
        if "def generate_submission(" not in content:
            print("❌ FAIL: generate_submission function not found in Track1 solution")
            return False
        
        print("✅ PASS: Track1 solution has generate_submission function")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error reading Track1 solution: {e}")
        return False

def test_leaderboard_file():
    """Test che il file leaderboard esista"""
    print("🧪 Testing leaderboard file...")
    
    leaderboard_file = Path("leaderboard.md")
    if not leaderboard_file.exists():
        print("❌ FAIL: leaderboard.md not found")
        return False
    
    try:
        with open(leaderboard_file, 'r') as f:
            content = f.read()
        
        if "# 🏆 SIAE Hackathon Leaderboard" not in content:
            print("❌ FAIL: leaderboard.md has wrong format")
            return False
        
        print("✅ PASS: leaderboard.md exists and has correct format")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error reading leaderboard: {e}")
        return False

def test_complete_workflow():
    """Test del workflow completo"""
    print("🧪 Testing complete workflow...")
    
    try:
        # 1. Test evaluation con file di esempio
        print("   Testing evaluation with example submission...")
        result = subprocess.run([sys.executable, "evaluate_submissions.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ FAIL: Evaluation failed: {result.stderr}")
            return False
        
        # 2. Verifica che leaderboard sia stata aggiornata
        print("   Checking if leaderboard was updated...")
        with open("leaderboard.md", 'r') as f:
            leaderboard_content = f.read()
        
        if "Team Example" not in leaderboard_content:
            print("❌ FAIL: Leaderboard not updated with example team")
            return False
        
        print("✅ PASS: Complete workflow successful")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ FAIL: Workflow test timeout")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error in complete workflow: {e}")
        return False

def run_all_tests():
    """Esegui tutti i test del sistema"""
    print("🚀 SIAE Hackathon - Evaluation System Tests")
    print("=" * 60)
    
    tests = [
        ("Submissions Directory", test_submissions_directory),
        ("Evaluation Script", test_evaluation_script),
        ("Track1 Solution", test_track1_solution),
        ("Leaderboard File", test_leaderboard_file),
        ("Complete Workflow", test_complete_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ FAIL: Unexpected error: {e}")
            results.append((test_name, False))
    
    # Riassunto risultati
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for hackathon!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED! Please fix issues before hackathon.")
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test - solo test essenziali
        print("🏃 Running quick tests...")
        success = (test_submissions_directory() and 
                  test_evaluation_script() and 
                  test_leaderboard_file())
        
        if success:
            print("✅ Quick tests passed!")
        else:
            print("❌ Quick tests failed!")
        
        return success
    else:
        # Test completi
        return run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 