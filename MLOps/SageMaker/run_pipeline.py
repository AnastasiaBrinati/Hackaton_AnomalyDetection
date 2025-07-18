#!/usr/bin/env python3
"""
Script unico per eseguire l'intera pipeline MLOps:
1. Build immagine Docker
2. Push su ECR
3. Lancia training SageMaker
"""
import boto3
import sagemaker
from sagemaker.estimator import Estimator
import time
import subprocess
import sys

def run_command(command):
    """Esegue un comando nel terminale e gestisce errori."""
    try:
        print(f"🚀 Eseguo: {' '.join(command)}")
        # Usiamo Popen per vedere l'output in tempo reale
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip())
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        print(f"✅ Comando completato con successo.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Errore durante l'esecuzione del comando: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Errore inaspettato: {e}", file=sys.stderr)
        return False

def run_pipeline():
    """Esegue l'intera pipeline MLOps."""
    # Configurazione
    project_name = 'mlops-exercise'
    region = 'eu-west-1'
    
    # Sessioni AWS
    sagemaker_session = sagemaker.Session()
    boto_session = sagemaker_session.boto_session
    account_id = boto_session.client('sts').get_caller_identity()['Account']
    
    # Nomi risorse
    ecr_repository_name = project_name
    image_tag = 'latest'
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository_name}:{image_tag}"
    role_arn = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-20250627T103254" # Sostituisci se necessario
    
    print("="*60)
    print("🚀 Inizio Pipeline MLOps Completa 🚀")
    print("="*60)

    # --- 1. Build Immagine Docker ---
    print("\n--- Fase 1: Build Immagine Docker ---")
    if not run_command(["docker", "build", "-t", ecr_repository_name, "model/"]):
        sys.exit(1)

    # --- 2. Autenticazione e Push su ECR ---
    print("\n--- Fase 2: Push su ECR ---")
    # Login
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    subprocess.run(login_cmd, shell=True, check=True)
    
    # Tag e Push
    if not run_command(["docker", "tag", f"{ecr_repository_name}:{image_tag}", image_uri]):
        sys.exit(1)
    if not run_command(["docker", "push", image_uri]):
        sys.exit(1)

    # --- 3. Lancia Training SageMaker ---
    print("\n--- Fase 3: Training su SageMaker ---")
    s3_input_path = f's3://sagemaker-mlops-exercise-{account_id}/{project_name}/input'
    s3_output_path = f's3://sagemaker-mlops-exercise-{account_id}/{project_name}/output'
    
    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=s3_output_path,
        sagemaker_session=sagemaker_session,
        hyperparameters={'epochs': '10', 'batch_size': '64'}
    )
    
    job_name = f"{project_name}-training-{int(time.time())}"
    print(f"\nLancio job di training: {job_name}")
    
    estimator.fit({'train': s3_input_path}, job_name=job_name, wait=True)
    
    print("\n🎉 Training completato con successo!")
    print(f"Artifacts salvati in: {estimator.model_data}")
    print("="*60)
    print("✅ Pipeline MLOps Eseguita Correttamente! ✅")
    print("="*60)

if __name__ == "__main__":
    run_pipeline() 