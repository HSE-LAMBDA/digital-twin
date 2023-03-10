import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from digital_twin.data import PoolDataSchema, CacheDataSchema
from tqdm import tqdm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

validators = {
    "pools": PoolDataSchema,
    "cache": CacheDataSchema,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default="dataset/", help="Path to result (json file)."
    )
    args = parser.parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Data file {args.dataset} not found.")
    return args


def get_train_test_data(df, test_fraction=0.2, random_state=666):
    ids = df["id"]
    train_ids, test_ids = train_test_split(
        list(set(ids)), test_size=test_fraction, random_state=random_state
    )
    return df[ids.isin(train_ids)], df[ids.isin(test_ids)]


def save_df(df, path):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    args = parse_args()
    for file in tqdm(list(args.dataset.rglob("**/*.csv"))):
        df = pd.read_csv(file)
        validators[file.parent.name].validate(df)
        train_df, test_df = get_train_test_data(df)

        save_df(train_df, file.parent / f"train_{file.name}")
        save_df(test_df, file.parent / f"test_{file.name}")
        logger.info(f"Saved {file.name}")
