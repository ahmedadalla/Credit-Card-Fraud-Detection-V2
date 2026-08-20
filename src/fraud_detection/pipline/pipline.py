from imblearn.pipeline import Pipeline
from src.fraud_detection.features.engineering import feature_engineering
from src.fraud_detection.data.preprocessing import preprocessor


def make_pipeline(sampler,model):
    return Pipeline([
        ("features", feature_engineering()),
        ("preprocessing", preprocessor),
        ("sampling", sampler),
        ("model", model)
    ])