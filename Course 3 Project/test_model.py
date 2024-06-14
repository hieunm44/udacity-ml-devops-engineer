import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data


@pytest.fixture(scope='module')
def data():
    return pd.read_csv('data/census_cleaned.csv')


@pytest.fixture(scope="module")
def cat_features():
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"]
    return cat_features


@pytest.fixture(scope='module')
def training_data(data, cat_features):
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True)
    return X_train, y_train


def test_train_model(training_data):
    X, y = training_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics(training_data):
    X, y = training_data
    model = pickle.load(open('model/model.pkl', 'rb'))
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(training_data):
    X, y = training_data
    model = pickle.load(open('model/model.pkl', 'rb'))
    preds = inference(model, X)
    assert y.shape == preds.shape and np.unique(preds).tolist() == [0, 1]