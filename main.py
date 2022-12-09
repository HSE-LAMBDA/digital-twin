import pandas as pd
import argparse
from pathlib import Path
from src.data import DataSchema

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data (csv file).')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    DataSchema.validate(df)
    print(df.info)
    
if __name__ == '__main__':
    main()
