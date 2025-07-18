#!/usr/bin/env python3
"""
MLOps SageMaker Setup - Versione Locale
Conversione del notebook Colab per uso locale

Autor: AI Assistant
Descrizione: Script per configurare l'ambiente MLOps SageMaker in locale
"""

import os
import sys
import json
import time
import platform
from pathlib import Path

# Gestione delle dipendenze con supporto TensorFlow
def install_and_import(package_name, import_name=None):
    """Installa e importa un pacchetto se non è disponibile"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} già installato")
        return True
    except ImportError:
        print(f"📦 Installando {package_name}...")
        import subprocess
        
        # Gestione speciale per TensorFlow
        if package_name == "tensorflow":
            return install_tensorflow()
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installato con successo")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Errore installazione {package_name}: {e}")
            return False

def install_tensorflow():
    """Installa TensorFlow con gestione di architetture diverse"""
    import subprocess
    
    print("🔍 Rilevamento architettura sistema...")
    
    # Rileva architettura
    system = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()
    
    print(f"Sistema: {system}")
    print(f"Architettura: {machine}")
    print(f"Python: {python_version}")
    
    # Verifica compatibilità Python
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 8 or minor > 11:
        print(f"❌ Python {python_version} non supportato da TensorFlow")
        print("💡 TensorFlow richiede Python 3.8-3.11")
        return False
    
    # Strategie di installazione per architettura
    strategies = []
    
    if system == "Darwin" and machine == "arm64":  # Apple Silicon
        strategies = [
            ("TensorFlow Metal (Apple Silicon)", ["tensorflow-macos==2.11.0"]),
            ("TensorFlow standard", ["tensorflow==2.11.0"]),
        ]
    elif machine in ["x86_64", "AMD64"]:  # Intel/AMD
        strategies = [
            ("TensorFlow con versione specifica", ["tensorflow==2.11.0"]),
            ("TensorFlow CPU-only", ["tensorflow-cpu==2.11.0"]),
            ("TensorFlow latest", ["tensorflow"]),
        ]
    else:  # ARM64/altre architetture
        strategies = [
            ("TensorFlow CPU-only", ["tensorflow-cpu==2.11.0"]),
            ("TensorFlow standard", ["tensorflow==2.11.0"]),
        ]
    
    # Prova ogni strategia
    for strategy_name, packages in strategies:
        print(f"🔄 Tentativo: {strategy_name}")
        try:
            # Aggiorna pip prima di ogni tentativo
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Installa pacchetti
            for package in packages:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "--no-cache-dir", package
                ])
            
            # Testa import
            import tensorflow as tf
            print(f"✅ TensorFlow {tf.__version__} installato con successo!")
            
            # Test aggiuntivo per Apple Silicon
            if system == "Darwin" and machine == "arm64":
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "tensorflow-metal"
                    ])
                    print("✅ TensorFlow Metal installato per accelerazione GPU")
                except:
                    print("⚠️ TensorFlow Metal non installato (opzionale)")
            
            return True
            
        except Exception as e:
            print(f"❌ Strategia '{strategy_name}' fallita: {e}")
            continue
    
    # Se tutte le strategie falliscono
    print("❌ Impossibile installare TensorFlow")
    print("\n💡 Soluzioni alternative:")
    print("1. Aggiorna Python alla versione 3.8-3.11")
    print("2. Crea un virtual environment pulito")
    print("3. Usa conda invece di pip:")
    print("   conda install tensorflow")
    print("4. Per Apple Silicon:")
    print("   conda install -c apple tensorflow-deps")
    print("5. Installa solo le dipendenze essenziali senza TensorFlow")
    
    return False

def install_essential_packages():
    """Installa solo i pacchetti essenziali (senza TensorFlow)"""
    print("📦 Installazione pacchetti essenziali...")
    
    essential_packages = [
        ("python-dotenv", "dotenv"),
        ("boto3", "boto3"),
        ("sagemaker", "sagemaker"),
        ("numpy", "numpy"),
    ]
    
    success_count = 0
    for package_name, import_name in essential_packages:
        if install_and_import(package_name, import_name):
            success_count += 1
    
    print(f"✅ {success_count}/{len(essential_packages)} pacchetti essenziali installati")
    
    # Prova TensorFlow come ultimo
    print("\n🔬 Tentativo installazione TensorFlow...")
    tf_success = install_and_import("tensorflow", "tensorflow")
    
    if not tf_success:
        print("⚠️ TensorFlow non installato - continuo senza")
        print("💡 Puoi installarlo manualmente più tardi")
        
        # Crea mock di TensorFlow per evitare errori
        try:
            import tensorflow as tf
        except ImportError:
            print("🔄 Creando mock TensorFlow temporaneo...")
            class MockTF:
                __version__ = "not_installed"
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            
            sys.modules['tensorflow'] = MockTF()
    
    return True

# Installa dipendenze con gestione errori migliorata
print("🔧 Verifica e installazione dipendenze...")
install_essential_packages()

# Ora importa le librerie
try:
    import boto3
    import sagemaker
    from sagemaker.estimator import Estimator
    from dotenv import load_dotenv
    import numpy as np
    
    # Import TensorFlow opzionale
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} disponibile")
    except ImportError:
        print("⚠️ TensorFlow non disponibile - alcune funzionalità saranno limitate")
        # Crea mock per evitare errori
        class MockTF:
            __version__ = "not_installed"
        tf = MockTF()
        
except ImportError as e:
    print(f"❌ Errore importazione critica: {e}")
    print("💡 Prova a installare manualmente:")
    print("   pip install boto3 sagemaker python-dotenv numpy")
    sys.exit(1)

# Resto del codice rimane uguale...
class MLOpsSetupLocal:
    """Classe per gestire il setup MLOps locale"""
    
    def __init__(self, env_file='.env'):
        """Inizializza il setup locale"""
        self.env_file = env_file
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.role = None
        self.aws_region = None
        self.account_id = None
        
        # Configurazione progetto
        self.project_name = 'mlops-fashion-classifier'
        self.ecr_repository_name = self.project_name
        self.image_tag = 'latest'
        
        # Sessioni AWS
        self.sagemaker_session = None
        self.boto_session = None
        
        # Nomi classi Fashion MNIST
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        print("🏠 MLOps Setup Locale - Inizializzazione")
        print("=" * 50)
    
    def load_environment_variables(self):
        """Carica le variabili d'ambiente dal file .env"""
        print("📁 Caricamento variabili d'ambiente...")
        
        # Carica file .env se esiste
        if Path(self.env_file).exists():
            load_dotenv(self.env_file)
            print(f"✅ File {self.env_file} caricato")
        else:
            print(f"⚠️  File {self.env_file} non trovato, usando variabili di sistema")
        
        # Carica variabili d'ambiente
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.role = os.getenv('SAGEMAKER_ROLE_ARN')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
        
        # Validazione
        if not all([self.aws_access_key_id, self.aws_secret_access_key, self.role]):
            missing_vars = []
            if not self.aws_access_key_id:
                missing_vars.append('AWS_ACCESS_KEY_ID')
            if not self.aws_secret_access_key:
                missing_vars.append('AWS_SECRET_ACCESS_KEY')
            if not self.role:
                missing_vars.append('SAGEMAKER_ROLE_ARN')
            
            error_msg = f"Variabili d'ambiente mancanti: {', '.join(missing_vars)}"
            print(f"❌ Errore: {error_msg}")
            print(f"💡 Assicurati di aver configurato il file {self.env_file} o le variabili di sistema")
            raise ValueError(error_msg)
        
        print("✅ Variabili d'ambiente caricate correttamente")
    
    def setup_aws_credentials(self):
        """Configura le credenziali AWS (equivalente al codice Colab)"""
        print("🔐 Configurazione credenziali AWS...")
        
        # Crea directory ~/.aws se non esiste
        aws_dir = Path.home() / '.aws'
        aws_dir.mkdir(exist_ok=True)
        
        # Scrive file credentials
        credentials_file = aws_dir / 'credentials'
        with open(credentials_file, 'w') as f:
            f.write(f"[default]\n")
            f.write(f"aws_access_key_id = {self.aws_access_key_id}\n")
            f.write(f"aws_secret_access_key = {self.aws_secret_access_key}\n")
        
        # Scrive file config
        config_file = aws_dir / 'config'
        with open(config_file, 'w') as f:
            f.write(f"[default]\n")
            f.write(f"region = {self.aws_region}\n")
        
        print("✅ Credenziali AWS configurate in ~/.aws/")
    
    def initialize_aws_sessions(self):
        """Inizializza le sessioni AWS (equivalente al codice Colab)"""
        print("🔗 Inizializzazione sessioni AWS...")
        
        # Le sessioni ora leggono la configurazione automaticamente
        self.sagemaker_session = sagemaker.Session()
        self.boto_session = self.sagemaker_session.boto_session
        
        # Ottieni account ID
        sts_client = self.boto_session.client('sts')
        identity = sts_client.get_caller_identity()
        self.account_id = identity['Account']
        
        print("✅ Sessioni AWS inizializzate")
    
    def setup_project_resources(self):
        """Configura le risorse del progetto"""
        print("📦 Configurazione risorse progetto...")
        
        # Nomi delle risorse
        self.bucket_name = f'sagemaker-fashion-mnist-{self.account_id}'
        
        print("-" * 50)
        print(f"🆔 Account AWS ID: {self.account_id}")
        print(f"🌍 Regione AWS: {self.aws_region}")
        print(f"🔑 Ruolo IAM: {self.role}")
        print(f"📦 Bucket S3: {self.bucket_name}")
        print(f"🐳 Nome Repository ECR: {self.ecr_repository_name}")
        print("-" * 50)
        
        return {
            'account_id': self.account_id,
            'aws_region': self.aws_region,
            'role': self.role,
            'bucket_name': self.bucket_name,
            'project_name': self.project_name,
            'ecr_repository_name': self.ecr_repository_name,
            'image_tag': self.image_tag,
            'class_names': self.class_names
        }
    
    def validate_aws_connection(self):
        """Valida la connessione AWS"""
        print("🧪 Validazione connessione AWS...")
        
        # Verifica che le sessioni siano inizializzate
        if not self.boto_session:
            print("❌ Sessione boto3 non inizializzata")
            return False
        
        if not self.role:
            print("❌ Ruolo SageMaker non configurato")
            return False
        
        try:
            # Test connessione
            sts_client = self.boto_session.client('sts')
            identity = sts_client.get_caller_identity()
            
            print(f"✅ Connessione AWS valida:")
            print(f"   User ARN: {identity['Arn']}")
            print(f"   Account: {identity['Account']}")
            
            # Test ruolo SageMaker
            iam_client = self.boto_session.client('iam')
            role_name = self.role.split('/')[-1]
            role_info = iam_client.get_role(RoleName=role_name)
            
            print(f"✅ Ruolo SageMaker valido: {role_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore validazione AWS: {e}")
            return False
    
    def setup_complete(self):
        """Esegue il setup completo"""
        try:
            # 1. Carica variabili d'ambiente
            self.load_environment_variables()
            
            # 2. Configura credenziali AWS
            self.setup_aws_credentials()
            
            # 3. Inizializza sessioni AWS
            self.initialize_aws_sessions()
            
            # 4. Configura risorse progetto
            config = self.setup_project_resources()
            
            # 5. Valida connessione
            if not self.validate_aws_connection():
                raise Exception("Validazione AWS fallita")
            
            print("\n🎉 Setup completato con successo!")
            print("🚀 Ora puoi procedere con il resto del progetto MLOps")
            
            return config
            
        except Exception as e:
            print(f"❌ Errore durante il setup: {e}")
            raise

def create_env_template():
    """Crea un template del file .env"""
    env_template = """# MLOps SageMaker - Configurazione Locale
# SOSTITUISCI CON I TUOI VALORI REALI

# AWS Credentials
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_HERE
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY_HERE
AWS_DEFAULT_REGION=eu-west-1

# SageMaker Role ARN
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX

# Project Configuration (opzionale)
PROJECT_NAME=mlops-fashion-classifier
TRAINING_INSTANCE_TYPE=ml.g4dn.xlarge
INFERENCE_INSTANCE_TYPE=ml.t2.medium
EPOCHS=5
BATCH_SIZE=128
"""
    
    if not Path('.env').exists():
        with open('.env', 'w') as f:
            f.write(env_template)
        print("📝 File .env template creato")
        print("⚠️  IMPORTANTE: Modifica il file .env con le tue credenziali AWS")
        return True
    else:
        print("📁 File .env già esistente")
        return False

def main():
    """Funzione principale"""
    print("🚀 MLOps SageMaker Setup - Versione Locale")
    print("=" * 60)
    
    # Crea template .env se non esiste
    if create_env_template():
        print("\n❗ Prima di continuare:")
        print("1. Modifica il file .env con le tue credenziali AWS")
        print("2. Esegui nuovamente lo script")
        return None, None
    
    # Esegui setup
    try:
        setup = MLOpsSetupLocal()
        config = setup.setup_complete()
        
        # Ritorna setup e config per uso esterno
        return setup, config
        
    except Exception as e:
        print(f"\n❌ Setup fallito: {e}")
        print("\n💡 Suggerimenti:")
        print("1. Verifica che il file .env sia configurato correttamente")
        print("2. Controlla che le credenziali AWS siano valide")
        print("3. Assicurati che il ruolo SageMaker esista")
        sys.exit(1)

if __name__ == "__main__":
    # Esegui setup quando script è chiamato direttamente
    setup, config = main()
    
    # Mostra informazioni finali solo se setup è riuscito
    if setup and config:
        print("\n📋 Configurazione disponibile:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        print("\n🎯 Prossimi passi:")
        print("1. Procedi con la preparazione dei dati")
        print("2. Crea gli script Docker (train.py, serve.py)")
        print("3. Build e push dell'immagine Docker")
        print("4. Avvia il training SageMaker") 