import subprocess
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--results-path', type=Path, default='results/predictions_data_size_sweep/', help='Path to store experiment results')
args = parser.parse_args()

if __name__ == '__main__':
    data_ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 0.1]

    for data_ratio in data_ratios:
        command = f'python scripts/denesity_estimation_experiment.py --data-ratio {data_ratio} --result-path={args.results_path / str(data_ratio)}'

        subprocess.run(command, shell=True)