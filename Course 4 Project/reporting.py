import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

deployed_model_path = os.path.join(config['prod_deployment_path'], "trainedmodel.pkl")
test_data_path = os.path.join(config['test_data_path'], "testdata.csv") 


##############Function for reporting
def score_model(model_path, data_path):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    model = pickle.load(open(model_path, 'rb'))
    df = pd.read_csv(data_path)
    X_test = df.drop(['corporation', 'exited'], axis='columns')
    y_test = df['exited']
    preds = model.predict(X_test)
    cf_matrix = metrics.confusion_matrix(y_test, preds)
    sns.heatmap(cf_matrix, annot=True)
    # plt.show()
    plt.savefig(os.path.join(config["output_model_path"], "confusionmatrix2.png"))  


if __name__ == '__main__':
    score_model(deployed_model_path, test_data_path)