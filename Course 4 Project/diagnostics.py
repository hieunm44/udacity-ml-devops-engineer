import os
import timeit
import json
import pandas as pd
import pickle
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')


##################Function to get model predictions
def model_predictions(model_path, data_path):
    #read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(model_path, 'rb'))
    df_test = pd.read_csv(data_path)
    X_test = df_test.drop(['corporation', 'exited'], axis='columns')
    preds = model.predict(X_test)    

    return preds #return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary(data_path):
    #calculate summary statistics here
    df = pd.read_csv(data_path)
    X = df.drop(['corporation', 'exited'], axis='columns')
    means = X.mean()
    medians = X.median()
    std = X.std()

    stats = {}
    for col in X.columns:
        stats[col] = {'means': means[col], 'medians': medians[col], 'std': std[col]}

    df_stats = pd.DataFrame(stats)

    return df_stats #return value should be a list containing all summary statistics


##################Function to Missing Data
def missing_data(data_path):
    df = pd.read_csv(data_path)
    na_list = list(df.isna().sum(axis=0))
    na_percents = [na/len(df.index) for na in na_list]
    
    return na_percents # return a list of percents of NA values


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    tic = timeit.default_timer()
    os.system('python3 ingestion.py')
    toc = timeit.default_timer()
    ingestion_time = toc - tic

    tic = timeit.default_timer()
    os.system('python3 training.py')
    toc = timeit.default_timer()
    training_time = toc - tic

    return [ingestion_time, training_time] #return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list():
    #get a list of current and latest versions of modules used in this script
    with open('requirements.txt', 'r') as f:
        current_modules = f.read().splitlines()
    
    modules = {'module_name': [], 'current_version': [], 'latest_version': []}
    for module in current_modules:
        module_name, current_version = module.split('==')
        latest_version = subprocess.check_output(['pip', 'index', 'versions', module_name])
        latest_version = latest_version.split(b'versions: ')[1].split(b', ')[0]
        latest_version = latest_version.decode('utf8')
        modules['module_name'].append(module_name)
        modules['current_version'].append(current_version)
        modules['latest_version'].append(latest_version)

    df_modules = pd.DataFrame(modules)

    return df_modules


if __name__ == '__main__':
    preds = model_predictions(deployed_model_path, test_data_path)
    # print('Model predictions:', preds)
    df_stats = dataframe_summary(test_data_path)
    # print('Data statistics:\n', df_stats)
    na_percents = missing_data(test_data_path)
    # print('Percentage of missing values:', na_percents)
    ingestion_time, training_time = execution_time()
    # print('Ingestion time:', ingestion_time, 'Training time:', training_time)
    df_modules = outdated_packages_list()
    # print('Dependencies:\n', df_modules)