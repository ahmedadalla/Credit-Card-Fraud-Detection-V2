from sklearn.base import BaseEstimator , TransformerMixin
import numpy as np

class feature_engineering(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X['Hour_of_Day'] = (X['Time'] // 3600) % 24
        X['Hour_sin'] = np.sin(2 * np.pi * X['Hour_of_Day'] / 24)
        X['Hour_cos'] = np.cos(2 * np.pi * X['Hour_of_Day'] / 24)
        return X