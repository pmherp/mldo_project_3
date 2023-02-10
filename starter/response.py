from fastapi import FastAPI, Response
import requests
import pandas as pd


print('url set')
url = "https://deploy-machine-learning-model-on-render.onrender.com/inference"
    
# Define the data to be sent in the POST request
print('data set')
data = {
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
}

#data = pd.read_csv('data/census.csv')
#data = data.drop('salary', axis=1)
#data = data.to_json(orient='columns')[0]
    
# Send the POST request to the Render API endpoint
print('response set')
response = requests.post(url, json=data)

# Check the status code of the response
print('status code set')
if response.status_code != 200:
    print(f"error: Failed to send POST request to Render API. Response status code: {response.status_code}")
    
# Try to get the result of model inference
print('result set')
try:
    result = response.json()
except ValueError as e:
    print(f"error: Failed to parse response content as JSON. Error: {e}")
    

# Print the result
print(f"result: {result}")
print(f"status_code: {response.status_code}")