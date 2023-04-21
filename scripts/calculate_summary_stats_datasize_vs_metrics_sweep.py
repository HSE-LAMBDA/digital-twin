from collections import defaultdict
import subprocess
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import groupby

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--results-path', type=Path, default='results/predictions_data_size_sweep/',
    help='Path to store experiment results'
)
parser.add_argument(
        '--metrics-path', type=Path, default='results/metrics_predictions_data_size_sweep/',
        help='Path to load metrics files'
)
args = parser.parse_args()

if __name__ == '__main__':
    
    for file in args.results_path.rglob('*.csv'):
        command = f"""python scripts/calculate_summary_stats.py --results {file.parent.parent} --make-plots True --save-path results/metrics_predictions_data_size_sweep/{file.parent.parent.name}"""

        subprocess.run(command, shell=True)
    metrics_files = list(args.metrics_path.rglob('metrics_agg_*.csv'))
    configs = ['cache', 'hdd_sequential', 'ssd_sequential', 'ssd_random']     
    
    groups = defaultdict(list)
    for conf in configs:
        for file in metrics_files:
            if conf in file.name:
                groups[conf].append(file)
                
    for group, files in groups.items():
        save_dir = args.results_path / f'datasize_vs_metrics_{group}'
        save_dir.mkdir(exist_ok=True, parents=True)
        x = [file.parents[2].name for file in files]
        y = [pd.read_csv(file, index_col=0) for file in files]
        
        y_iops = [df.loc['MEAPE_IOPS'].loc['mean'] for df in y]
        y_lat = [df.loc['MEAPE_LAT'].loc['mean'] for df in y] 
        
        
        plt.plot(x[::-1], y_iops[::-1], color='darkred')
        plt.xlabel('Training data fraction')
        plt.ylabel('MEAPE')
        plt.title('IOPS MEAPE')
        plt.savefig(save_dir / 'meape_iops.pdf')
        plt.clf()
        
        plt.plot(x[::-1], y_lat[::-1], color='darkred')
        plt.xlabel('Training data fraction')
        plt.ylabel('MEAPE')
        plt.title('IOPS MEAPE')
        plt.savefig(save_dir / 'meape_lat.pdf')
        plt.clf()
        
       
        
    