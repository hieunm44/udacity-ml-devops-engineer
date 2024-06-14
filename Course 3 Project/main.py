# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import inference


class InputData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example='State-gov') 
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example='Bachelors')
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example='Never-married')
    occupation: str = Field(..., example='Adm-clerical')
    relationship: str = Field(..., example='Not-in-family')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Male')
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example='United-States')


app = FastAPI()


@app.get('/')
async def welcome():
    return "Welcome to our service!"


@app.post('/predict/')
async def predict(input_data: InputData):
    data_dict  = {key.replace('_', '-'): value for key, value in input_data.dict().items()}
    data = pd.DataFrame(data_dict, index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    model = pickle.load(open('model/model.pkl', 'rb'))
    encoder = pickle.load(open('model/encoder.pkl', 'rb'))
    lb = pickle.load(open('model/lb.pkl', 'rb'))

    X, _, _, _ = process_data(
                    data,
                    categorical_features=cat_features,
                    label=None,
                    training=False,
                    encoder=encoder,
                    lb=lb)

    preds = inference(model, X)
    res = '>50K' if preds[0]==1 else '<=50K'

    return {'prediction': res}