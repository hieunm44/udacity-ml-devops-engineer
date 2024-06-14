import requests
import json


sample = {"age": 50,
          "workclass": "Self-emp-not-inc",
          "fnlgt": 83311,
          "education": "Bachelors",
          "education_num": 13,
          "marital_status": "Married-civ-spouse",
          "occupation": "Exec-managerial",
          "relationship": "Husband",
          "race": "White",
          "sex": "Male",
          "capital_gain": 0,
          "capital_loss": 0,
          "hours_per_week": 13,
          "native_country": "United-States"
          }

r = requests.post(url='https://salary-prediction-from-census-data.onrender.com/predict/', data=json.dumps(sample))
print('Status code:', r.status_code)
print('Results:', r.json())