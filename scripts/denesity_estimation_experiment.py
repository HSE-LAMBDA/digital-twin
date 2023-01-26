import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from digital_twin.data import PoolDataSchema
from digital_twin.models.density_estimation import Grouper
from digital_twin.models.density_estimation import GMM
from digital_twin.data import POOL_DEPENDENT_VARS, POOL_INDEPENDENT_VARS
from digital_twin.models.density_estimation import Regressor
from logging import getLogger
from functools import partial
from sklearn.model_selection import train_test_split
from digital_twin.performance_metrics.mmd import mmd_rbf

logger = getLogger(__name__)
logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to data (csv file)."
    )
    parser.add_argument(
        "--model-checkpoint-path",
        type=Path,
        default="models_checkpoints",
        help="Path to model (cbm file).",
        
    )
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    if not args.model_checkpoint_path.exists():
        args.model_checkpoint_path.mkdir(parents=True)
    return args


def train_test_train_predict(fn, X, y, indices, **kwargs):

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.3, random_state=42)
    
    model = fn(X_train, y_train, X_test, y_test)
    params = model.predict(X_test)
    params = pd.DataFrame(params, columns=y_test.columns)
    
    df = []
    for idx, (_, row) in zip(indices_test, params.iterrows()):
        gmm = GMM()
        gmm.init_from_params(
            weights=row.filter(regex="weight").values,
            means=row.filter(regex="mean").values.reshape(
                gmm.model.n_components, -1),
            precisions_cholesky=row.filter(regex="precision").values.reshape(
                gmm.model.n_components, -1)
        )
        generated_samples = np.expm1(gmm.sample(len(idx)))
        chunk_df = (
            kwargs.get('target_df').loc[idx]
                .assign(
                    **{
                        'generated_iops': generated_samples[:, 0],
                        'generated_lat': generated_samples[:, 1]
                    }
            )
        )
        df.append(chunk_df)
    return pd.concat(df), indices_test
    
    
def fit(x, y, x_test, y_test, model_checkpoint_path):
    regressor = Regressor(
        regressor_params=dict(
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            iterations=5000,
            learning_rate=0.1,
            early_stopping_rounds=None,
            random_seed=42,
            use_best_model=True,
            metric_period=100,
            # depth=6,
            # l2_leaf_reg=3,
            # border_count=128,
        ),
    )
    models = regressor.fit(x, y, eval_set=(x_test, y_test))
    regressor.save_model(model_checkpoint_path)
    return models

def metrics_eval(target: np.array, prediction: np.array):
    metrics = {"mmd_rbf": mmd_rbf(target, prediction)}
    return metrics

def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    PoolDataSchema.validate(df)
    
    logger.info(df.info)
    x, y, grouped_indices, shapes = Grouper().transform(
        df, dependent_vars=POOL_DEPENDENT_VARS, independent_vars=POOL_INDEPENDENT_VARS
    )
    
    test_set_predictions, test_set_indices = train_test_train_predict(
        partial(
            fit, 
            model_checkpoint_path=args.model_checkpoint_path / f"{args.data.stem}.cbm"
        ),
        x, y, indices=grouped_indices, shapes=shapes, target_df=df
    )
    metrics = metrics_eval(test_set_predictions[['iops', 'lat']].values, test_set_predictions[['generated_iops', 'generated_lat']].values)
    logger.info(metrics)
    # metrics.to_csv(args.model_checkpoint_path / f"{args.data.stem}_metrics.csv", index=False)
    


if __name__ == "__main__":
    main()
