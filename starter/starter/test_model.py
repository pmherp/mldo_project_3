import pytest
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
# from ml.model import load_model


@pytest.fixture
def data():
    data = pd.read_csv('../data/census.csv')
    y = data['salary']
    X = data.drop(['salary'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_train_model(data):
    X_train, _, y_train, _ = data
    with open("../model/model.pkl", "rb") as model_file:
        clf = pickle.load(model_file)

    assert clf.classes_ is not None, 'Classes not found'
    assert clf.n_classes_ == len(
        np.unique(y_train)), 'Incorrect number of classes'


def test_model_parameters(data):
    X_train, _, y_train, _ = data
    with open("../model/model.pkl", "rb") as model_file:
        clf = pickle.load(model_file)

    assert clf.n_estimators == 100, 'Incorrect number of trees'


def test_data_types():
    data = pd.read_csv('../data/census.csv')

    expected_types = {
        "Unnamed: 0.4": int,
        "Unnamed: 0.3": int,
        "Unnamed: 0.2": int,
        "Unnamed: 0.1": int,
        "Unnamed: 0": int,
        "age": int,
        "workclass": object,
        "fnlgt": int,
        "education": object,
        "education-num": int,
        "marital-status": object,
        "occupation": object,
        "relationship": object,
        "race": object,
        "sex": object,
        "capital-gain": int,
        "capital-loss": int,
        "hours-per-week": int,
        "native-country": object,
        "salary": object
    }

    for column, expected_type in expected_types.items():
        assert data[column].dtype == expected_type, f'Incorrect dtype for column {column}'
