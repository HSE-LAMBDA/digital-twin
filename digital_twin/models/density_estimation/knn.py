from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from digital_twin.performance_metrics.misc import absolute_percentage_error as ape
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd

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
                 categories=['load_type', 'io_type', 'device_type'], params_grid={'n_neighbors': [1], 'p': [2]}):
                 #params_grid={'n_neighbors': [1, 2, 5, 7, 10, 15, 20], 'p': [1, 2]}):
        self.column_transformer = YadroColumnsTransformer(columns_to_drop)
        self.model = Pipeline([
            ('encoding', OneHotEncoder(sparse=False, handle_unknown='infrequent_if_exist')),
            ('scaling', StandardScaler()),
            ('model', GridSearchCV(KNeighborsRegressor(),
                               param_grid=params_grid,
                               scoring=make_scorer(scoring_fn)))
        ])


    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self._orig_features = list(X.columns)
        X = self.column_transformer.transform(X)
        self._transformed_features = list(X.columns)
        self._targets = list(y.columns)
        df_grouped = pd.concat([X, y], axis=1).groupby(self._transformed_features).apply(lambda df: df.mean())[self._targets].reset_index(drop=False)
        X_grouped, y_grouped = df_grouped[self._transformed_features], df_grouped[self._targets]
        self._X = X
        self._y = y
        self._X_grouped = X_grouped
        self._y_grouped = y_grouped
        self.model.fit(X_grouped, y_grouped)

    def sample(self, n_samples:int, **configuration):
        X_test = np.array([configuration[k] for k in self._orig_features])
        X_test = np.expand_dims(X_test, 0)
        X_test_transformed = pd.DataFrame(X_test, columns=self._orig_features).transform(pd.to_numeric, errors='ignore', downcast='float')
        X_test_transformed = self.column_transformer.transform(X_test_transformed)
        if isinstance(self.model, Pipeline):
            for name, transform in self.model.steps[:-1]:
                X_test_transformed = transform.transform(X_test_transformed)
            neigh_inds = self.model.steps[-1][-1].best_estimator_.kneighbors(X_test_transformed, return_distance=False)
        else:
            neigh_inds = self.model.kneighbors(X_test_transformed, return_distance=False)
        df = pd.concat([self._X, self._y], axis=1)
        neigh_inds = np.reshape(neigh_inds, (-1))
        neigh_df = self._X_grouped.iloc[neigh_inds]
        samples_dfs = []
        
        for _, row in neigh_df.iterrows():
            samples_df = df
            for feat in self._transformed_features:
                samples_df = samples_df[samples_df[feat] == row[feat]]
            samples_dfs.append(samples_df)
        samples_df = pd.concat(samples_dfs).sample(n_samples).reset_index(drop=True)
        samples_df = self.column_transformer.inverse_transform(samples_df)
        return samples_df
#         return samples_df[self._features], samples_df[self._targets]
