"""
Contains a function to make batch or single prediction and
writes predictions into a file and returns them as output
"""

from starter.ml.data import process_data
from starter.ml.model import inference, load_model
import pickle5 as pickle


def inference_model(data, cat_features):
    """
    Loads the model and runs the inference

    Parameters
    ----------
    root_path
    data
    cat_features

    Returns
    -------
    prediction
    """
    try:
        trained_model, encoder, lb = load_model(
            "/Users/philipherp/Documents/Udacity/Machine_Learning_DevOps/mldo_project_3/starter/model/model.pkl",
            "/Users/philipherp/Documents/Udacity/Machine_Learning_DevOps/mldo_project_3/starter/model/encoder.pkl",
            "/Users/philipherp/Documents/Udacity/Machine_Learning_DevOps/mldo_project_3/starter/model/lb.pkl")
    except:
        with open("model/model.pkl", "rb") as model_file:
            trained_model = pickle.load(model_file)
        
        with open("model/encoder.pkl", "rb") as model_file:
            encoder = pickle.load(model_file)

        with open("model/lb.pkl", "rb") as model_file:
            lb = pickle.load(model_file)

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)

    pred = inference(trained_model, X)

    prediction = lb.inverse_transform(pred)[0]

    return prediction
