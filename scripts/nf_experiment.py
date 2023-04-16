import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from digital_twin.models.norm_flow.model import NormFlowModel, DEFAULT_CHECKPOINTS_DIR
from sklearn.model_selection import train_test_split
import torch
import random
import numpy as np
from glob import glob
import pandas as pd
import os, sys

from sklearn.metrics import make_scorer
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from digital_twin.performance_metrics import ape

logger = logging.getLogger(__name__)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# save logs to file and console
if not Path("logs").exists():
    Path("logs").mkdir(parents=True)
logger.addHandler(logging.FileHandler("logs/nf_experiment.log"))


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


def match_files(train_files, test_files):
    train_files_new = []
    test_files_new = []
    for atrain in train_files:
        train_files_new.append(atrain)
        for atest in test_files:
            if str(atrain).split("train_")[-1] == str(atest).split("test_")[-1]:
                test_files_new.append(atest)
                break
    return train_files_new, test_files_new


def scoring_fn(y_true, y_pred):
    return -ape(y_true.values, y_pred).mean()


def get_X_y(df):
    df[['r0', 'r1']] = df['raid'].str.split('+', n=1, expand=True)
    df.replace({'random': 0, 'sequential': 1, 'read': 0, 'write': 1}, inplace=True)
    return df.drop(["iops", "lat", "id", 'raid', 'device_type', 'offset'], axis=1), df[["iops", "lat"]]


def get_predictions(train_df, test_df, model_checkpoint_path):
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)
    
    logger.info(f"Starting training. Model`s checkpoint dir: {model_checkpoint_path}")
    
    model = NormFlowModel(model_checkpoint_path.__str__())
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1337)
    model.fit(X_tr, y_tr, X_val, y_val, n_epochs=80)
    
    logger.info('Acquire predictions')
    preds = []
    for i in tqdm(range(X_test.shape[0])):
        inst = X_test.iloc[i]
        pred = model.sample(X=inst)
        preds.append([pred[:, 0], pred[:, 1]])
        
    return np.array(preds)


def main(train_file, test_file):
    print("Train: %s \nTest: %s \n" % (train_file, test_file))
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    (root_dir := args.model_checkpoint_path / f"{train_file.parent.name}" / f"nf_{train_file.stem}").mkdir(
        exist_ok=True
    )

    this_prediction = get_predictions(
        train_df,
        test_df,
        model_checkpoint_path=root_dir)
    test_df = test_df.assign(
        gen_iops=this_prediction[:, 0], gen_lat=this_prediction[:, 1]
    )

    (root_dir := args.result_path / f"{train_file.parent.name}").mkdir(exist_ok=True, parents=True)
    test_df.to_csv(root_dir /  f"nf_pred_{test_file.name}", index=False)
    logger.info(f"Saving predictions: {train_file.parent.name} - {test_file.name}")


if __name__ == "__main__":
    args = parse_args()
    train_files = list(args.data.rglob("**/train_*.csv"))
    test_files = list(args.data.rglob("**/test_*.csv"))
    train_files, test_files = match_files(train_files, test_files)
    for train, test in tqdm(zip(train_files, test_files)):
        main(train, test)