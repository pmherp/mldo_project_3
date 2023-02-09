"""
This script tests the API.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """
    Get data
    """
    data = TestClient(app)

    return data


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Message": "Hello there! --> General Kenobi!"}


def test_get_wrong_url(client):
    r = client.get("/wrong_url")
    assert r.status_code != 200


def test_post_above_50(client):
    r = client.post("/inference", json={
        "age": 30,
        "workclass": "State-gov",
        "fnlgt": 180211,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "India"
    })
    assert r.status_code == 200, 'Status Code not 200.'
    assert r.json() == {"prediction": "<=50K"}, f'Prediction wrong: {r.json()}'


def test_post_below_50(client):
    r = client.post("/inference", json={
        "age": 19,
        "workclass": "Private",
        "fnlgt": 0,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 99999,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    assert r.status_code == 200, 'Status Code not 200.'
    assert r.json() == {"prediction": "<=50K"}, f'Prediction wrong: {r.json()}'
