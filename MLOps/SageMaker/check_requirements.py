#!/usr/bin/env python3
"""
Script per verificare che tutti i prerequisiti per il test script siano soddisfatti
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Verifica la versione di Python"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version OK (3.8+)")
        return True
    else:
        print("❌ Python version troppo vecchia. Richiesto: Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """Verifica se un package è installato"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            print(f"✅ {package_name} installed")
            return True
        else:
            print(f"❌ {package_name} NOT installed")
            return False
    except ImportError:
        print(f"❌ {package_name} NOT installed")
        return False

def check_aws_credentials():
    """Verifica le credenziali AWS"""
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        print(f"✅ AWS credentials configured (Account: {account_id})")
        return True
    except Exception as e:
        print(f"❌ AWS credentials NOT configured: {e}")
        print("💡 Run: aws configure")
        return False

def check_lambda_function():
    """Verifica se esistono funzioni Lambda del progetto"""
    try:
        import boto3
        lambda_client = boto3.client('lambda')
        
        response = lambda_client.list_functions()
        project_keywords = ['mlops', 'fashion', 'classifier', 'exercise', 'invoker']
        
        found_functions = []
        for function in response['Functions']:
            function_name = function['FunctionName'].lower()
            for keyword in project_keywords:
                if keyword in function_name:
                    found_functions.append(function['FunctionName'])
                    break
        
        if found_functions:
            print(f"✅ Lambda functions found: {', '.join(found_functions)}")
            return True
        else:
            print("⚠️  No Lambda functions found (test remoto non disponibile)")
            return False
            
    except Exception as e:
        print(f"❌ Cannot check Lambda functions: {e}")
        return False

def main():
    """Funzione principale"""
    print("🔍 === VERIFICA PREREQUISITI TEST SCRIPT ===")
    print("=" * 50)
    
    all_ok = True
    
    # Verifica Python
    if not check_python_version():
        all_ok = False
    
    print("\n📦 Verifica Packages Python:")
    required_packages = [
        ('tensorflow', 'tensorflow'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('boto3', 'boto3'),
    ]
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_ok = False
    
    print("\n🔧 Verifica Configurazione AWS:")
    aws_ok = check_aws_credentials()
    lambda_ok = check_lambda_function()
    
    if not aws_ok:
        all_ok = False
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("🎉 ✅ TUTTI I PREREQUISITI SODDISFATTI!")
        print("🚀 Puoi eseguire il test script con:")
        print("   python test_model_predictions.py")
        
        if lambda_ok:
            print("\n🌐 Test remoto disponibile (AWS Lambda)")
        else:
            print("\n⚠️  Test remoto non disponibile (solo test locale)")
    else:
        print("❌ ALCUNI PREREQUISITI MANCANO")
        print("\n🔧 Azioni da eseguire:")
        print("1. Installa packages mancanti:")
        print("   pip install -r requirements.txt")
        print("   pip install matplotlib boto3")
        print("2. Configura AWS se necessario:")
        print("   aws configure")
        print("3. Riprova la verifica:")
        print("   python check_requirements.py")

if __name__ == "__main__":
    main() 