import pandas as pd
import argparse
from pathlib import Path
from src.data import PoolDataSchema
from src.models.catboost.group import Group
from src.data import POOL_DEPENDENT_VARS, POOL_INDEPENDENT_VARS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data (csv file).')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    PoolDataSchema.validate(df)
    print(df.info)
    Group(df, dependent_vars=POOL_DEPENDENT_VARS, independent_vars=POOL_INDEPENDENT_VARS)
    breakpoint()
    
if __name__ == '__main__':
    main()
