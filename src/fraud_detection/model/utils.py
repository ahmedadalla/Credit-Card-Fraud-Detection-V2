from src.fraud_detection.configs import Paths
from src.fraud_detection.configs.config import config
import joblib

MODELS=Paths.MODELS
def save_model(model, name):
    path = MODELS / name

    with open(path, "wb") as file:
        joblib.dump(model, file)

def load_model(name):
    path = MODELS / name
    return joblib.load(path)

def get_model_params(name):
    if name=='lg':
        return config.model.logistic_regression
    elif name == 'rf':
        return config.model.random_forest
    elif name == 'nn':
        return config.model.mlp
    elif name=="knn":
        return config.model.knn

def set_model_params(model, params):
    model.set_params(**params)
    return model
