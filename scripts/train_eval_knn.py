from glob import glob
import pandas as pd
import os
import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.data import PoolDataSchema, CacheDataSchema
from src.utils.metrics import mean_estimation_absolute_percentage_error as meape
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import make_scorer
import json
import pickle


RANDOM_STATE = 42
RESULT_PATH = 'output/knn.json'
CHECKPOINTS_PATH = 'checkpoints/knn_%s.pkl'
TEST_FRACTION = 0.03

if __name__ == '__main__':
    result = {}

    for filepath in glob('dataset/**', recursive=True):
        if not filepath.endswith('.csv'): continue
        df = pd.read_csv(filepath)
        if 'pool' in filepath:
            schema = PoolDataSchema
            filename = '_'.join(filepath.split('/')[-3:-1])
        elif filepath.endswith('cache_data.csv'):
            schema = CacheDataSchema
            filename = 'cache'
        schema.validate(df)
        model = Pipeline([
            ('encoding', OneHotEncoder(sparse=False)),
            ('scaling', StandardScaler()),
            ('model', GridSearchCV(KNeighborsRegressor(), 
                                   param_grid={
                                       'n_neighbors': [2, 5, 7, 9, 10, 20],
                                       'p': [1, 2, 3]
                                   },
                                   scoring=make_scorer(lambda y_true, y_pred: -meape(y_true.values, y_pred)[0].mean())))
        ])
        X, y, ids = df.drop(['iops', 'lat', 'id'], 1), df[['iops', 'lat']], df['id']
        train_ids, test_ids = train_test_split(list(set(ids)), test_size=TEST_FRACTION, random_state=RANDOM_STATE)
        X_train, X_test = X[ids.isin(train_ids)], X[ids.isin(test_ids)]
        y_train, y_test = y[ids.isin(train_ids)], y[ids.isin(test_ids)]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        meape_iops_mean, meape_iops_std = meape(y_test.values[:, 0], y_pred[:, 0])
        meape_lat_mean, meape_lat_std = meape(y_test.values[:, 1], y_pred[:, 1])
        with open(CHECKPOINTS_PATH%filename, 'wb') as f:
            pickle.dump(model, f)
        result[filename] = {'IOPS': (meape_iops_mean, meape_iops_std), 'LAT': (meape_lat_mean, meape_lat_std)}
        print(f'MEAPE_IOPS: {meape_iops_mean:.2f}±{meape_iops_std:.2f}')
        print(f'MEAPE_LAT: {meape_lat_mean:.2f}±{meape_lat_std:.2f}')
    with open(RESULT_PATH, 'w') as f:
        f.write(json.dumps(result, sort_keys=True, indent=4))
