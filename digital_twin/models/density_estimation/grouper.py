import pandas as pd
from pandas import DataFrame
from dataclasses import asdict
from digital_twin.data import PoolDataSchema
from .gmm import GMM
import numpy as np
from tqdm import tqdm


class  Grouper:
    """Group parameters."""
    
    def transform(self, df, dependent_vars: list[str], independent_vars: list[str]):
        self.df = df
        self.groupby = list(set(PoolDataSchema.__fields__.keys()) - set(dependent_vars))
        groups = self.df.groupby(self.groupby)

        # keep track of the indices of the groups useful for performance evaluation
        grouped_indices = []
        x = []
        y = []
        shapes = None
        for _, (name, group) in tqdm(enumerate(groups)):
            grouped_indices.append(group.index)
            gmm = GMM(n_components=2, covariance_type="full", use_elliptic_envelope=True) 
            gmm.fit(group.loc[:, dependent_vars].apply(np.log1p))
            _params = dict(
                precision = gmm.precisions_cholesky,
                means = gmm.means,
                weights = gmm.weights,
            )
            # Add numerical suffix to keys and flaten the list
            shapes = {k: v.shape for k, v in _params.items()}
            _params = {f'{k}_{i}': vv for k, v in _params.items() for i, vv in enumerate(v.reshape(-1)) }
            targets = pd.DataFrame(_params, index=[0])
            targets = targets.reindex(sorted(targets.columns), axis=1)
            x.append(self.featurize(group.loc[:, independent_vars].drop_duplicates()))
            y.append(targets)
            # if _ == 2:
            #     break
        x = pd.concat(x)
        y = pd.concat(y)
        return x, y, grouped_indices, shapes
    
    def featurize(self, df):
        """Featurize the data."""
        df = pd.concat(
            [
                df,
                df.raid.apply(lambda x: list(map(int, x.split("+"))))
                .apply(pd.Series)
                .rename({0: "n_disk", 1: "n_parity_disk"}, axis=1),
            ],
            axis=1,
        )
        return df.assign(**{
            "bs/n_disk": lambda x: x.block_size / x.n_disks,
            "total_disk": lambda x: x.n_parity_disk + x.n_disk,
            "iodepth_jobs": df.iodepth * df.n_jobs
            }
        )