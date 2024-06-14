import os
import json
from scoring import score_model


with open('config.json','r') as f:
    config = json.load(f)


input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']

dataset_csv_path = os.path.join(output_folder_path, 'finaldata.csv')
ingested_file_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
deployed_model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
latest_score = os.path.join(prod_deployment_path, 'latestscore.txt')


##################Check and read new data
#first, read ingestedfiles.txt
with open(ingested_file_path, 'r') as f:
    ingested_files = f.read().splitlines()


#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
has_new_data = False
for filename in os.listdir(input_folder_path):
    if filename not in ingested_files:
        has_new_data = True
        break


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if has_new_data:
    os.system('python3 ingestion.py')
else:
    exit(0)


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
has_model_drift = False
with open(latest_score, 'r') as f:
    latest_score = float(f.read())
new_score = score_model(deployed_model_path, dataset_csv_path)
if new_score < latest_score:
    has_model_drift = True


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if has_model_drift:
    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    os.system('python3 training.py')
    os.system('python3 scoring.py')
    os.system('python3 deployment.py')

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    os.system('python3 diagnostics.py')
    os.system('python3 reporting.py')
    os.system('python3 apicalls.py')