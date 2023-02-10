# Put the code for your API here.

import os
import uvicorn
from fastapi import FastAPI
from features import Features
import pandas as pd
from starter.inference import inference_model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")

app = FastAPI()


@app.get("/")
async def root():
    return {"Message": "Hello there! --> General Kenobi!"}


@app.post("/inference")
async def inference(input_data: Features):
    input_data = input_data.dict()

    columns = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education_num',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital_gain',
        'capital_loss',
        'hours_per_week',
        'native_country'
    ]

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    input_df = pd.DataFrame(
        data=input_data.values(),
        index=input_data.keys()
    ).T

    input_df = input_df[columns]

    try:
        data = pd.read_csv('data/census.csv')
        data = data.drop('salary', axis=1)
    except:
        data = pd.read_csv('starter/data/census.csv')
        data = data.drop('salary', axis=1)

    prediction = inference_model(data, cat_features)

    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="https://deploy-machine-learning-model-on-render.onrender.com", port=8080)
