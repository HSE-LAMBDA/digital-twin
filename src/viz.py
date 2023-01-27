import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import similaritymeasures
from sklearn.preprocessing import StandardScaler
from ssd_sim.utils import metrics, fd


def basic_plots(y_true, y_pred, title="", data=None):
    
    iops_true = y_true[:, 0]
    iops_pred = y_pred[:, 0]
    
    lat_true = y_true[:, 1]/10**6
    lat_pred = y_pred[:, 1]/10**6
    
    plt.figure(figsize=(21, 5))
    
    plt.subplot(131)
    plt.scatter(iops_true, lat_true, alpha=0.5, marker='o', label='True')
    plt.scatter(iops_pred, lat_pred, alpha=0.5, marker='+', label='Prediction')
    plt.xlabel('IOPS', size=14)
    plt.ylabel('Latency, ms', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=14)
    
    plt.subplot(132)
    vals = np.concatenate((iops_true, iops_pred))
    bins = np.linspace(vals.min(), vals.max(), 50)
    plt.hist(iops_true, bins=bins, alpha=1., label='True', histtype='step', linewidth=3)
    plt.hist(iops_pred, bins=bins, alpha=1., label='Prediction')
    plt.xlabel('IOPS', size=14)
    plt.ylabel('Counts', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=14)
    
    plt.subplot(133)
    vals = np.concatenate((lat_true, lat_pred))
    bins = np.linspace(vals.min(), vals.max(), 50)
    plt.hist(lat_true, bins=bins, alpha=1., label='True', histtype='step', linewidth=3)
    plt.hist(lat_pred, bins=bins, alpha=1., label='Prediction')
    plt.xlabel('Latency, ms', size=14)
    plt.ylabel('Counts', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title, size=14)
    plt.legend(loc='best', fontsize=14)

    cols = list(data.head(1).columns)
    values = data.head(1).values[0].tolist()
    text = ''.join([f'{col}: {val}. ' for (col, val) in zip(cols, values)])

    plt.figtext(0.5, 0.96, text, ha="center", va="center",
                fontsize=14)

    plt.savefig(f'{title}.png')

    plt.show()


def test_pipeline(model, test, result_dir, random_test=None):
    """

    Args:
        model: SSDModel with loaded checkpoints
        test: Pandas DataFrame with test data
        result_dir: a directory to contain graphs of evaluated model
        random_test: Integer. Randomly picking specified amount of test runs for evaluation.

    Returns:
        Pandas DataFrame with metrics, plots graphs

    """
    report = []
    x_cols = ['iodepth', 'block_size', 'read_fraction', 'io_type', 'load_type', 'n_jobs']
    y_cols = ['iops', 'latency']

    test_runs = test['run'].unique()
    test_runs = np.random.choice(test_runs, size=random_test) if random_test else test_runs

    for run in test_runs:
        trun = test[test['run'] == run]

        for io_type in [0, 1]:
            trun_io = trun[trun['io_type'] == io_type]

            if len(trun_io) == 0: continue

            cond = trun_io[x_cols]
            y_test = trun_io[y_cols].values

            y_pred_test = model.sample(n_samples=y_test.shape[0], iodepth=cond['iodepth'].values[0],
                                       block_size=cond['block_size'].values[0],
                                       read_fraction=cond['read_fraction'].values[0],
                                       io_type=cond['io_type'].values[0],
                                       load_type=cond['load_type'].values[0],
                                       n_jobs=cond['n_jobs'].values[0])
            
            

            title = 'Write'
            if io_type == 0:
                title = 'Read'
            if not os.path.isdir(result_dir):
                os.mkdir(result_dir)
            if result_dir[-1] != '/':
                result_dir = f'{result_dir}/'
            basic_plots(y_test, y_pred_test, title=result_dir + run+'__'+title, data=cond)

            # metrics
            report_run = []

            mu, std = metrics.mean_estimation_absolute_percentage_error(y_test, y_pred_test, n_iters=100)
            m, s = fd.bootstrap_frdist(y_test, y_pred_test, n_iters=100)            
            print(f'Frechet distance-2nd_opt: {m}+-{s}')
            print(r"IOPS    MEAPE: %.2f +- %.2f %%" % (mu[0], std[0]))
            print(r"Latency MEAPE: %.2f +- %.2f %%" % (mu[1], std[1]))
            report_run += [mu[0], mu[1]]

            report.append(report_run)

        print("\n\n")

    report = pd.DataFrame(columns=["IOPS_MEAPE", "Lat_MEAPE"], data=report)
    print("Mean metrics values for all test runs: ")
    print(report.mean())
    return report
