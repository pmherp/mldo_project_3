"""
This module outputs the performance of the model on slices of the data for categorical features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, load_model

def performance_on_metrics():
    """ Check performance on categorical features """

    df = pd.read_csv("../data/census.csv")

    _, test = train_test_split(df, test_size=0.2)

    trained_model, encoder, lb = load_model("../model/model.pkl", "../model/encoder.pkl", "../model/lb.pkl")

    features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=features,
        label="salary", encoder=encoder, lb=lb, training=False)

    y_preds = trained_model.predict(X_test)

    print(X_test.shape)

    prc, rcl, fb = compute_model_metrics(y_test, y_preds)

    print('Precision Score : ',prc)
    print('Recall Score : ',rcl)
    print('FBeta Score : ', fb)

if __name__ == '__main__':
    performance_on_metrics()