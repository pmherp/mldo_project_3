from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred

def save_model(model, encoder, lb):
    """ Saves the model and encoder

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : OneHotEncoder

    Returns
    -------
    None
    """
    with open("../model/model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("../model/encoder.pkl", "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)
    
    with open("../model/lb.pkl", "wb") as lb_file:
        pickle.dump(lb, lb_file)

def load_model(model_pth, encoder_pth, lb_pth):
    """ Loads the model and encoder

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : OneHotEncoder

    Returns
    -------
    model : RandomForestClassifier
    encoder : OneHotEncoder
    """
    with open(model_pth, "rb") as model_file:
        model = pickle.load(model_file)
    
    with open(encoder_pth, "rb") as encoder_file:
        encoder = pickle.load(encoder_file)
    
    with open(lb_pth, "rb") as lb_file:
        lb = pickle.load(lb_file)
    
    return model, encoder, lb
