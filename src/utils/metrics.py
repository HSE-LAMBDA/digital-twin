import numpy as np
import pandas as pd
import os, sys
import re
from sklearn.utils import resample
import pdb

def absolute_percentage_error(y_true, y_pred):
    eps = 1e-8
    return np.abs((y_true-y_pred)/(eps+y_true))*100

def mean_estimation_absolute_percentage_error(ids, y_true, y_pred, n_iters=100):   
    errors = []
    ids = np.array(ids)
    for id in set(ids):
        errors_id = []
        id_mask = ids == id
        inds = np.arange(id_mask.astype(int).sum())
        for i in range(n_iters):
            inds_boot = resample(inds)
            y_true_boot = y_true[np.argwhere(id_mask)][inds_boot]
            y_pred_boot = y_pred[np.argwhere(id_mask)][inds_boot]
            
            y_true_mean = y_true_boot.mean(0)
            y_pred_mean = y_pred_boot.mean(0)
            ierr = np.abs((y_true_mean - y_pred_mean) / y_true_mean) * 100
            errors_id.append(ierr.squeeze())
        errors.append(np.mean(errors_id, axis=0))
    return np.mean(errors, axis=0)

def std_estimation_absolute_percentage_error(ids, y_true, y_pred, n_iters=100, eps=1e-8):   
    errors = []
    ids = np.array(ids)
    for id in set(ids):
        errors_id = []
        id_mask = ids == id
        inds = np.arange(id_mask.astype(int).sum())
        for i in range(n_iters):
            inds_boot = resample(inds)
            y_true_boot = y_true[np.argwhere(id_mask)][inds_boot]
            y_pred_boot = y_pred[np.argwhere(id_mask)][inds_boot]
        
            y_true_std = y_true_boot.std(0)
            y_pred_std = y_pred_boot.std(0)
            ierr = np.abs((y_true_std - y_pred_std) / (eps+y_true_std)) * 100
            errors_id.append(ierr.squeeze())
        errors.append(np.mean(errors_id, axis=0))
        
    return np.mean(errors, axis=0)
