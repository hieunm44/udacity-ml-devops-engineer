import requests
import os
import json


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

#Call each API endpoint and store the responses
response1 = requests.post(URL + 'prediction', data={'data_path': test_data_path}) #put an API call here
response2 = requests.get(URL + 'scoring') #put an API call here
response3 = requests.get(URL + 'summarystats') #put an API call here
response4 = requests.get(URL + 'diagnostics') #put an API call here

#combine all API responses
responses = {
    'prediction': response1.json(),
    'scoring': response2.json(),
    'summarystats': response3.json(),
    'diagnostics': response4.json()
} #combine reponses here

#write the responses to your workspace
with open(os.path.join(config["output_model_path"], 'apireturns2.txt'), 'w') as f:
    json.dump(responses, f)