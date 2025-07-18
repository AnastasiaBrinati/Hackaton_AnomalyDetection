import json
import boto3
import os
import sagemaker

ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Nomi delle classi per una risposta più chiara
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def lambda_handler(event, context):
    try:
        # L'input è nel body, ci aspettiamo: {"image_data": [...]}
        body = json.loads(event['body'])
        image_data = body['image_data']

        # Il modello si aspetta un payload JSON con la chiave "instances"
        payload = {"instances": [image_data]}

        # Invoca l'endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        # Legge il risultato
        result = json.loads(response['Body'].read().decode())

        # Estrae la classe predetta e il nome della classe
        predicted_class_index = result['predictions'][0]
        predicted_class_name = class_names[predicted_class_index]

        # Risposta finale
        final_response = {
            'predicted_class_index': predicted_class_index,
            'predicted_class_name': predicted_class_name
        }

        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps(final_response)
        }

    except Exception as e:
        print(f"Errore: {e}")
        return { 'statusCode': 500, 'body': json.dumps({'error': str(e)}) }

import zipfile
import time

project_name = 'mlops-exercise'



# Crea il client Lambda
sagemaker_session = sagemaker.Session()
project_name = 'mlops-exercise'

boto_session = sagemaker_session.boto_session
iam_client = boto_session.client('iam')
lambda_client = boto_session.client('lambda')

boto_session = sagemaker_session.boto_session

# Creazione del pacchetto di deploy
zip_path = 'mlops_fashion_mnist/lambda_function.zip'
with zipfile.ZipFile(zip_path, 'w') as zf:
    zf.write('mlops_fashion_mnist/lambda_function/lambda_function.py', arcname='lambda_function.py')

# Nomi
lambda_function_name = f'{project_name}-invoker'
lambda_role_name = f'{project_name}-lambda-role'

iam_client = boto_session.client('iam')
lambda_client = boto_session.client('lambda')

# 1. Creazione del ruolo IAM per la Lambda
assume_role_policy = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
try:
    role_response = iam_client.create_role(RoleName=lambda_role_name, AssumeRolePolicyDocument=json.dumps(assume_role_policy))
    lambda_role_arn = role_response['Role']['Arn']
    print(f"Ruolo IAM '{lambda_role_name}' creato.")
    iam_client.attach_role_policy(RoleName=lambda_role_name, PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole')
    iam_client.attach_role_policy(RoleName=lambda_role_name, PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess')
    print("Policy allegate. Attendo 10s per la propagazione...")
    time.sleep(10)
except iam_client.exceptions.EntityAlreadyExistsException:
    print(f"Ruolo IAM '{lambda_role_name}' già esistente.")
    lambda_role_arn = iam_client.get_role(RoleName=lambda_role_name)['Role']['Arn']

# 2. Creazione della funzione Lambda
with open(zip_path, 'rb') as f: zipped_code = f.read()
try:
    lambda_client.create_function(
        FunctionName=lambda_function_name, Runtime='python3.8', Role=lambda_role_arn,
        Handler='lambda_function.lambda_handler', Code={'ZipFile': zipped_code},
        Environment={'Variables': {'SAGEMAKER_ENDPOINT_NAME': predictor.endpoint_name}},
        Timeout=15
    )
    print(f"Funzione Lambda '{lambda_function_name}' creata.")
except lambda_client.exceptions.ResourceConflictException:
    print(f"Funzione Lambda '{lambda_function_name}' già esistente. La aggiorno.")
    lambda_client.update_function_code(FunctionName=lambda_function_name, ZipFile=zipped_code)
    lambda_client.update_function_configuration(FunctionName=lambda_function_name, Environment={'Variables': {'SAGEMAKER_ENDPOINT_NAME': predictor.endpoint_name}})
     