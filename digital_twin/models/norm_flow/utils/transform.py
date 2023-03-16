from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from dataclasses import dataclass

class LogScaler(TransformerMixin, BaseEstimator):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log(X+self.eps)
    
    def inverse_transform(self, X):
        return np.exp(X)-self.eps