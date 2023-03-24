import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm.contrib.concurrent import process_map

from digital_twin.models.density_estimation import GMM, Grouper, Regressor

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# save logs to file and console
if not Path("logs").exists():
    Path("logs").mkdir(parents=True)
logger.addHandler(logging.FileHandler("logs/density_estimation_experiment.log"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default="dataset", help="Path to data (csv file)."
    )
    parser.add_argument(
        "--model-checkpoint-path",
        type=Path,
        default="models_checkpoints",
        help="Path to model (cbm file).",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default="results/predictions",
        help="Path to result (json file).",
    )
    parser.add_argument(
        "--grid-search",
        type=bool,
        default=False,
        help="Run grid search.",
    )
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    if not args.model_checkpoint_path.exists():
        args.model_checkpoint_path.mkdir(parents=True)
    return args


def get_X_y(df):
    return df.drop(["iops", "lat", "id"], axis=1), df[["iops", "lat"]]

def preprocess_X(df):
    df.load_type = df.load_type.factorize()[0]
    df.io_type = df.io_type.factorize()[0]
    df = df.drop(['offset', 'raid', 'device_type'], axis=1)
    return df

def regressor(
    X_train, y_train, X_test, y_test, model_checkpoint_path, grid_search=False
):
    regressor = Regressor(
        regressor_params=dict(
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            iterations=5000,
            learning_rate=0.1,
            early_stopping_rounds=None,
            random_seed=42,
            metric_period=100,
            use_best_model=True,
            allow_const_label=True,
        ),
    )
    grid_search_params = {
        "depth": [2, 4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    regressor.fit(
        X_train,
        y_train,
        grid_search_params=grid_search_params if grid_search else None,
        eval_set=(X_test, y_test),
    )
    regressor.save_model(model_checkpoint_path)
    return regressor


def init_gmm_params(gmm, row):
    gmm.init_from_params(
        weights=row.filter(regex="weight").values,
        means=row.filter(regex="mean").values.reshape(gmm.model.n_components, -1),
        precisions_cholesky=row.filter(regex="precision").values.reshape(
            gmm.model.n_components, -1
        ),
    )


def get_predictions(
    train_df,
    test_df,
    model_checkpoint_path,
    grid_search=False,
):
    X_train, y_train, _, _ = Grouper.transform(*get_X_y(train_df))
    X_test, y_test, test_indices, _ = Grouper.transform(*get_X_y(test_df))
    X_train = preprocess_X(X_train)
    X_test = preprocess_X(X_test)
    model = regressor(
        X_train, y_train, X_test, y_test, model_checkpoint_path, grid_search=grid_search
    )
    params = pd.DataFrame(model.predict(X_test), columns=y_test.columns)
    df = []

    for idx, (_, row) in zip(test_indices, params.iterrows()):
        gmm = GMM(n_components=1)
        init_gmm_params(gmm, row)
        generated_samples = gmm.sample(len(idx))
        chunk_df = test_df.loc[idx].assign(
            gen_iops=generated_samples[:, 0],
            gen_lat=generated_samples[:, 1],
        )
        df.append(chunk_df)
    return pd.concat(df).loc[test_df.index]


def main(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    (root_dir := args.model_checkpoint_path / f"{train_file.parent.name}").mkdir(
        exist_ok=True
    )
    predictions = get_predictions(
        train_df,
        test_df,
        root_dir / f"catboost_{train_file.stem}.cbm",
        grid_search=True if args.grid_search else False,
    )

    (root_dir := args.result_path / f"{train_file.parent.name}").mkdir(
        exist_ok=True, parents=True
    )
    predictions.to_csv(
        root_dir / f"catboost_density_pred_{test_file.name}", index=False
    )
    logger.info(f"Saving predictions: {train_file.parent.name} - {test_file.name}")


if __name__ == "__main__":
    args = parse_args()
    train_files = list(args.data.rglob("train_*.csv")) 
    test_files = list(args.data.rglob("test_*.csv"))

    process_map(main, train_files, test_files)