from glob import glob
import pandas as pd
import os
import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from digital_twin.data import PoolDataSchema, CacheDataSchema
from functools import partial
from digital_twin.performance_metrics.eape import absolute_percentage_error as ape, mean_estimation_absolute_percentage_error as meape, std_estimation_absolute_percentage_error as seape, aggregate_loads
from sklearn.model_selection import train_test_split
from digital_twin.models.knn import KNNSampler
import json
import pickle
import pdb


RANDOM_STATE = 42
RESULTS_DIR = 'outputs/'
CHECKPOINTS_DIR = 'checkpoints/'
if not os.path.isdir(RESULTS_DIR): os.mkdir(RESULTS_DIR)
if not os.path.isdir(CHECKPOINTS_DIR): os.mkdir(CHECKPOINTS_DIR)
TEST_FRACTION = 0.03

if __name__ == '__main__':
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
            scaler = StandardScaler()
            y_true = scaler.fit_transform(y_true)
            y_pred = scaler.transform(y_pred)
            return -ape(y_true, y_pred).mean()

        model = KNNSampler()
        X, y, ids = df.drop(['iops', 'lat', 'id'], 1), df[['iops', 'lat']], df['id']


        train_ids, test_ids = train_test_split(list(set(ids)), test_size=TEST_FRACTION, random_state=RANDOM_STATE)
        X_train, X_test = X[ids.isin(train_ids)], X[ids.isin(test_ids)]
        y_train, y_test = y[ids.isin(train_ids)], y[ids.isin(test_ids)]
        ids_train, ids_test = ids[ids.isin(train_ids)], ids[ids.isin(test_ids)]
        model.fit(X_train, y_train)
        test_configs = {test_id: X_test[ids_test == test_id].to_dict(orient='records')[0] for test_id in ids_test}
        def sampling_fn(col, sample_size=100, **kwargs):
            return model.sample(sample_size, **kwargs)[1][[col]].values

        

#        samples_X, samples_y = model.sample(n_samples=10,
#                                            block_size=128, n_jobs=1, iodepth=5, read_fraction=100,
#                                            load_type='sequential', io_type='read', raid='4+1', 
#                                            n_disks=10, device_type='hdd', offset=0)
        # model.predict(X_train_grouped, n_samples=2) # [X.shape, n_samples, n_targets]
        meape_iops_mean, meape_iops_std = aggregate_loads(ids_test.values, test_configs, y_test.values[:, 0], partial(sampling_fn, 'iops'), meape)
        meape_lat_mean, meape_lat_std = aggregate_loads(ids_test.values, test_configs, y_test.values[:, 0], partial(sampling_fn, 'lat'), meape)

        seape_iops_mean, seape_iops_std = aggregate_loads(ids_test.values, test_configs, y_test.values[:, 0], partial(sampling_fn, 'iops'), seape)
        seape_lat_mean, seape_lat_std = aggregate_loads(ids_test.values, test_configs, y_test.values[:, 0], partial(sampling_fn, 'lat'), seape)

        result = {'MEAPE_IOPS': {'mean': meape_iops_mean, 'std': meape_iops_std}, 'MEAPE_LAT': {'mean': meape_lat_mean, 'std': meape_lat_std},
                            'SEAPE_IOPS': {'mean': seape_iops_mean, 'std': seape_iops_std}, 'SEAPE_LAT': {'mean': seape_lat_mean, 'std': seape_iops_std}}
        print(f'MEAPE_IOPS: {meape_iops_mean:.2f} ± {meape_iops_std:.2f}')
        print(f'MEAPE_LAT: {meape_lat_mean:.2f} ± {meape_lat_std:.2f}')
        print(f'SEAPE_IOPS: {seape_iops_mean:.2f} ± {seape_iops_std:.2f}')
        print(f'SEAPE_LAT: {seape_lat_mean:.2f} ± {seape_lat_std:.2f}')

        results_path = os.path.join(RESULTS_DIR, filename, 'knn.json')
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, filename, 'knn.pkl')
        if not os.path.isdir(os.path.dirname(results_path)): os.mkdir(os.path.dirname(results_path))
        if not os.path.isdir(os.path.dirname(checkpoint_path)): os.mkdir(os.path.dirname(checkpoint_path))
        with open(results_path, 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4))
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
