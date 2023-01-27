from glob import glob
import pandas as pd
import os
import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.data import PoolDataSchema, CacheDataSchema
from src.utils.metrics import absolute_percentage_error as ape, mean_estimation_absolute_percentage_error as meape, std_estimation_absolute_percentage_error as seape
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
if not os.path.isdir('output'): os.mkdir('output')
if not os.path.isdir('checkpoints'): os.mkdir('checkpoints')
TEST_FRACTION = 0.03

if __name__ == '__main__':
    result = {}

    for filepath in glob('dataset/**', recursive=True):
        if not filepath.endswith('.csv'): continue
        df = pd.read_csv(filepath)
        if 'pool' in filepath:
            schema = PoolDataSchema
            filename = '_'.join(filepath.split('/')[-2:]).strip('.csv')
        elif filepath.endswith('cache_data.csv'):
            schema = CacheDataSchema
            filename = 'cache'
        schema.validate(df)
        def scoring_fn(y_true, y_pred):
            return -ape(y_true.values, y_pred).mean()
        model = Pipeline([
            ('encoding', OneHotEncoder(sparse=False)),
            ('scaling', StandardScaler()),
            ('model', GridSearchCV(KNeighborsRegressor(),
                                   param_grid={
                                       'n_neighbors': [2, 5, 7, 10, 15, 20],
                                       'p': [1, 2]
                                   },
                                   scoring=make_scorer(scoring_fn)))
        ])
        X, y, ids = df.drop(['iops', 'lat', 'id'], 1), df[['iops', 'lat']], df['id']
        train_ids, test_ids = train_test_split(list(set(ids)), test_size=TEST_FRACTION, random_state=RANDOM_STATE)
        X_train, X_test = X[ids.isin(train_ids)], X[ids.isin(test_ids)]
        y_train, y_test = y[ids.isin(train_ids)], y[ids.isin(test_ids)]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        meape_iops_mean, meape_iops_std = meape(ids[ids.isin(test_ids)].values, y_test.values[:, 0], y_pred[:, 0])
        meape_lat_mean, meape_lat_std = meape(ids[ids.isin(test_ids)].values, y_test.values[:, 1], y_pred[:, 1])
    
        seape_iops_mean, seape_iops_std = seape(ids[ids.isin(test_ids)].values, y_test.values[:, 0], y_pred[:, 0])
        seape_lat_mean, seape_lat_std = seape(ids[ids.isin(test_ids)].values, y_test.values[:, 1], y_pred[:, 1])
    
        with open(CHECKPOINTS_PATH%filename, 'wb') as f:
            pickle.dump(model, f)
        result[filename] = {'MEAPE_IOPS': {'mean': meape_iops_mean, 'std': meape_iops_std}, 'MEAPE_LAT': {'mean': meape_lat_mean, 'std': meape_lat_std},
                            'SEAPE_IOPS': {'mean': seape_iops_mean, 'std': seape_iops_std}, 'SEAPE_LAT': {'mean': seape_lat_mean, 'std': seape_iops_std}}
        print(f'MEAPE_IOPS: {meape_iops_mean:.2f} ± {meape_iops_std:.2f}')
        print(f'MEAPE_LAT: {meape_lat_mean:.2f} ± {meape_lat_std:.2f}')
        print(f'SEAPE_IOPS: {seape_iops_mean:.2f} ± {seape_iops_std:.2f}')
        print(f'SEAPE_LAT: {seape_lat_mean:.2f} ± {seape_lat_std:.2f}')


    with open(RESULT_PATH, 'w') as f:
        f.write(json.dumps(result, sort_keys=True, indent=4))
