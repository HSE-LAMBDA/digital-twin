import numpy as np
import pandas as pd
import os, sys
import re
from sklearn.utils import resample

def mean_estimation_absolute_percentage_error(y_true, y_pred, n_iters=100):
    
    errors = []
    
    inds = np.arange(len(y_true))
    for i in range(n_iters):
        inds_boot = resample(inds)
        
        y_true_boot = y_true[inds_boot]
        y_pred_boot = y_pred[inds_boot]
        
        y_true_mean = y_true_boot.mean(axis=0)
        y_pred_mean = y_pred_boot.mean(axis=0)
        
        ierr = np.abs((y_true_mean - y_pred_mean) / y_true_mean) * 100
        errors.append(ierr)
        
    errors = np.array(errors)
    return errors.mean(axis=0), errors.std(axis=0)

