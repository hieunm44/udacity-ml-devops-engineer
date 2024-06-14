import os
import pickle
import json
import pandas as pd
from sklearn.metrics import f1_score


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
output_model_path = config['output_model_path']
deployed_model_path = os.path.join(output_model_path, 'trainedmodel.pkl')


#################Function for model scoring
def score_model(model_path, data_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    model = pickle.load(open(model_path, 'rb'))
    df = pd.read_csv(data_path)
    X_test = df.drop(['corporation', 'exited'], axis='columns')
    y_test = df['exited']
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)

    if not os.path.isdir(output_model_path):
        os.mkdir(output_model_path)
        
    with open(os.path.join(output_model_path, "latestscore.txt"), 'w') as f:
        f.write(str(f1))
    
    return f1


if __name__ == '__main__':
    f1 = score_model(deployed_model_path, test_data_path)
    print('F1:', f1)