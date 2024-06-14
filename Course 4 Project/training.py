import os
import pickle
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression



###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl') 


#################Function for training the model
def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(['corporation', 'exited'], axis='columns')
    y = df['exited']

    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model.fit(X, y)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.exists(config['output_model_path']):
        os.mkdir(config['output_model_path'])

    pickle.dump(model, open(model_path, 'wb'))


if __name__ == '__main__':
    train_model(dataset_csv_path)