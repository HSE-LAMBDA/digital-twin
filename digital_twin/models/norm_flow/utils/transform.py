from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from dataclasses import dataclass

class LogScaler(TransformerMixin, BaseEstimator):
    def __init__(self, eps=1e-8, a_max=50):
        super().__init__()
        self.eps = eps
        self.a_max = a_max
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log(X+self.eps)
    
    def inverse_transform(self, X):
        X = np.clip(X, a_min=None, a_max=self.a_max)
        return np.exp(X)-self.eps