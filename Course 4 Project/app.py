from flask import Flask, request
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
# app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def prediction():        
    #call the prediction function you created in Step 3
    data_path = request.form.get('data_path')
    preds = model_predictions(deployed_model_path, data_path)

    return json.dumps(preds.tolist()) #add return value for prediction outputs


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    f1 = score_model(deployed_model_path, test_data_path)
    return json.dumps(f1) #add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    df_stats = dataframe_summary(dataset_csv_path)

    return json.dumps(df_stats.to_dict()) #return a list of all calculated summary statistics


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    ingestion_time, training_time = execution_time()
    na_percents = missing_data(dataset_csv_path)
    dependencies = outdated_packages_list().to_dict()
    diagnose_dict = {
        'ingestion_time': ingestion_time,
        'training_time': training_time,
        'na_percents': na_percents,
        'dependencies': dependencies
    }

    return json.dumps(diagnose_dict) #add return value for all diagnostics


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)