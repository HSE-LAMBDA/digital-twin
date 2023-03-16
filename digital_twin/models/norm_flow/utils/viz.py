import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def basic_plots(y_true, y_pred, title=""):
    
    iops_true = y_true[:, 0]
    iops_pred = y_pred[:, 0]
    
    lat_true = y_true[:, 1]/10**6
    lat_pred = y_pred[:, 1]/10**6
    
    plt.figure(figsize=(21, 4))
    
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
    
    plt.show()