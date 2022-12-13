import pandas as pd
from pandas import DataFrame
from dataclasses import asdict
from ...data import PoolDataSchema
from .gmm import GMM
import numpy as np

class Group:
    """Group parameters."""

    def __init__(self, df: DataFrame, dependent_vars: list[str], independent_vars: list[str]):
        self.df = df
        self.groupby = list(set(PoolDataSchema.__fields__.keys()) - dependent_vars)
        groups = self.df.groupby(self.groupby)

        # keep track of the indices of the groups useful for performance evaluation
        grouped_indices = []
        for name, group in groups:
            gmm = GMM(group.loc[:, dependent_vars].apply(np.log1p), n_components=2, covariance_type="full", use_elliptic_envelope=True) 
            grouped_indices.append(group.index)
            breakpoint()
