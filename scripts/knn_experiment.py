import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import make_scorer
from tqdm.contrib.concurrent import process_map
from digital_twin.models.density_estimation.knn import KNNSampler
import warnings; warnings.filterwarnings('ignore')

from digital_twin.performance_metrics import ape

logger = logging.getLogger(__name__)
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
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    if not args.model_checkpoint_path.exists():
        args.model_checkpoint_path.mkdir(parents=True)
    return args


def scoring_fn(y_true, y_pred):
    return -ape(y_true.values, y_pred).mean()


def get_X_y(df):
    return df.drop(["iops", "lat", "id", "device_type", "offset"], axis=1), df[["iops", "lat"]]

def get_predictions(train_df, test_df, model_checkpoint_path):
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)
    id_test = test_df.id

    model = KNNSampler()

    model.fit(X_train, y_train)
    with open(model_checkpoint_path, "wb") as f:
        pickle.dump(model, f)
        
    dfs = []
    for id_, conf, gt in zip(list(id_test), X_test.to_dict(orient='records'), y_test.to_dict(orient='records')):
        samples_df = model.sample(n_samples=1, **conf)
        samples_df['id'] = id_
        samples_df.rename(columns={'lat': 'gen_lat', 'iops': 'gen_iops'}, inplace=True)
        samples_df['iops'] = gt['iops']
        samples_df['lat'] = gt['lat']
        dfs.append(samples_df)
    res = pd.concat(dfs).reset_index(drop=True)
    return res

def main(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    (root_dir := args.model_checkpoint_path / f"{train_file.parent.name}").mkdir(
        exist_ok=True
    )

    test_df = get_predictions(
        train_df,
        test_df,
        model_checkpoint_path=root_dir / f"knn_{train_file.stem}.pkl",
    )

    (root_dir := args.result_path / f"{train_file.parent.name}").mkdir(exist_ok=True, parents=True)
    test_df.to_csv(root_dir /  f"knn_pred_{test_file.name}", index=False)
    logger.info(f"Saving predictions: {train_file.parent.name} - {test_file.name}")


if __name__ == "__main__":
    args = parse_args()
    train_files = list(args.data.rglob("**/train_*.csv"))
    test_files = list(args.data.rglob("**/test_*.csv"))
    process_map(main, train_files, test_files)
    #for train_file, test_file in zip(train_files, test_files):
    #    main(train_file, test_file)
