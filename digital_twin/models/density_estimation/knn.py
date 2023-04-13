from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from digital_twin.performance_metrics.misc import absolute_percentage_error as ape
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
import copy

def default_scoring_fn(y_true, y_pred):
    scaler = StandardScaler()
    y_true = scaler.fit_transform(y_true)
    y_pred = scaler.transform(y_pred)
    return -ape(y_true, y_pred).mean()

class YadroColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=['device_type']):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[[col for col in X.columns if col not in self.columns_to_drop]]
        X['raid_0'] = X.raid.apply(lambda x: x.split('+')[0]).astype(int)
        X['raid_1'] = X.raid.apply(lambda x: x.split('+')[1]).astype(int)
        X['load_type'] = X.load_type.apply(lambda x: 0 if x=='random' else 1)
        if y is None:
            return X.drop(['raid'], 1)
        return X.drop(['raid'], 1), y

    def inverse_transform(self, X, y=None):
        X = X.copy()
        X['raid'] = X['raid_0'].astype(str) + '+' + X['raid_1'].astype(str)
        X.drop(['raid_0', 'raid_1'], 1, inplace=True)
        if y is None:
            return X
        return X, y

class KNNSampler:
    def __init__(self, scoring_fn=default_scoring_fn, columns_to_drop=['device_type'],
                 categories=['load_type', 'io_type', 'device_type'], n_neighbors=1, p=2):
                 #params_grid={'n_neighbors': [1, 2, 5, 7, 10, 15, 20], 'p': [1, 2]}):
        self.column_transformer = YadroColumnsTransformer(columns_to_drop)
        model_builder = lambda: Pipeline([
            ('scaling', StandardScaler()),
            ('model', KNeighborsRegressor(n_neighbors=n_neighbors, p=p))
        ])
        self.model_read, self.model_write = [model_builder() for _ in range(2)]


    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self._orig_features = list(X.columns)
        X = self.column_transformer.transform(X)
        self._transformed_features = [col for col in X.columns if col not in {'io_type'}]
        self._targets = list(y.columns)
        for io_type in ['read', 'write']:
            X_ = X.query(f'io_type=="{io_type}"').drop('io_type', 1, inplace=False)
            y_ = y[X.io_type == io_type]
            df_grouped = pd.concat([X_, y_], axis=1).groupby(self._transformed_features).apply(lambda df: df.mean())[self._targets].reset_index(drop=False)
            X_grouped, y_grouped = df_grouped[self._transformed_features], df_grouped[self._targets]
            if io_type == 'read':
                self._X_read, self._y_read, self._X_grouped_read, self._y_grouped_read = X_, y_, X_grouped, y_grouped
                model = self.model_read
            else:
                self._X_write, self._y_write, self._X_grouped_write, self._y_grouped_write = X_, y_, X_grouped, y_grouped
                model = self.model_write
            if X_grouped.shape[0] > 0:
                model.fit(X_grouped, y_grouped)

    def sample(self, n_samples:int, **configuration):
        io_type = configuration['io_type']
        if io_type == 'read' and self._X_read.shape[0] == 0:
            io_type = 'write'
        elif io_type == 'write' and self._X_write.shape[0] == 0:
            io_type == 'read'
        X_test = np.array([configuration[k] for k in self._orig_features])
        X_test = np.expand_dims(X_test, 0)
        X_test_transformed = pd.DataFrame(X_test, columns=self._orig_features).transform(pd.to_numeric, errors='ignore', downcast='float')
        X_test_transformed.drop('io_type', 1, inplace=True)
        X_test_transformed = self.column_transformer.transform(X_test_transformed)
        model = {'read': self.model_read, 'write': self.model_write}[io_type]
        if isinstance(model, Pipeline):
            for name, transform in model.steps[:-1]:
                X_test_transformed = transform.transform(X_test_transformed)
            neigh_inds = model.steps[-1][-1].kneighbors(X_test_transformed, return_distance=False)
        else:
            neigh_inds = model.kneighbors(X_test_transformed, return_distance=False)
        X = {'read': self._X_read, 'write': self._X_write}[io_type]
        y = {'read': self._y_read, 'write': self._y_write}[io_type]
        df = pd.concat([X, y], axis=1)
        neigh_inds = np.reshape(neigh_inds, (-1))
        X_grouped = {'read': self._X_grouped_read, 'write': self._X_grouped_write}[io_type]
        neigh_df = X_grouped.iloc[neigh_inds]
        samples_dfs = []
        
        for _, row in neigh_df.iterrows():
            samples_df = df
            for feat in self._transformed_features:
                samples_df = samples_df[samples_df[feat] == row[feat]]
            samples_dfs.append(samples_df)
        samples_df = pd.concat(samples_dfs).sample(n_samples).reset_index(drop=True)
        samples_df = self.column_transformer.inverse_transform(samples_df)
        output_df = pd.DataFrame([configuration]*n_samples)
        output_df['iops'] = samples_df['iops']
        output_df['lat'] = samples_df['lat']
        return output_df
