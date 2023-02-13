import requests


features = {
        "age": 38,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Handlers-cleaners",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }


app_url = "https://deploy-machine-learning-model-on-render.onrender.com/inference"

r = requests.post(app_url, json=features)
assert r.status_code == 200

print(f"Status code: {r.status_code}")
print(f"Response body: {r.json()}")