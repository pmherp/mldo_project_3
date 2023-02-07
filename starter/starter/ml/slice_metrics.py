import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from data import process_data
from model import compute_model_metrics, load_model


def accuracy():
    """
    Execute accuracy
    """
    df = pd.read_csv("../../data/census.csv")
    _, test = train_test_split(df, test_size=0.20)

    # trained_model = load("model/model.joblib")
    # encoder = load("model/encoder.joblib")
    # lb = load("model/lb.joblib")

    trained_model, encoder, lb = load_model(
        "../../model/model.pkl", "../../model/encoder.pkl", "../../model/lb.pkl")

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

    slice_values = []

    for cat in features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=features,
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = compute_model_metrics(y_test, y_preds)

            line = "[%s on %s] Precision: %s | " \
                   "Recall: %s | FBeta: %s" % (cat, cls, prc, rcl, fb)

            slice_values.append(line)

    with open('../../model/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')


if __name__ == '__main__':
    accuracy()
