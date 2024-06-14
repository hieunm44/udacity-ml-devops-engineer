from fastapi.testclient import TestClient
from main import app
import json


client = TestClient(app)


def test_welcome():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == 'Welcome to our service!'


def test_predict_negative():
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
    
    r = client.post('/predict/', data=json.dumps(sample))
    assert r.status_code == 200
    assert r.json() == {'prediction': '<=50K'}


def test_predict_positive():
    sample = {"age": 42,
            "workclass": "Private",
            "fnlgt": 159449,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 5178,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States"
            }
    
    r = client.post('/predict/', data=json.dumps(sample))
    assert r.status_code == 200
    assert r.json() == {'prediction': '>50K'}