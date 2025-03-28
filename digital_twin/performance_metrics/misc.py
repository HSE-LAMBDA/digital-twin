import numpy as np
import pandas as pd
import os, sys
import re
from sklearn.utils import resample
import pdb

def absolute_percentage_error(y_true, y_pred):
    eps = 1e-8
    return np.abs((y_true-y_pred)/(eps+y_true))*100

def aggregate_loads(load_ids, y_true, y_pred, metric_fn):
    errors = []
    load_ids = np.array(load_ids)
    for id in set(load_ids):
        load_inds = np.argwhere(id == load_ids)
        errors.append(metric_fn(y_true[load_inds], y_pred[load_inds]))
    return np.mean([x[0] for x in errors]), np.sqrt(sum(map(lambda x: x[1]**2,errors))/len(errors))


def _eval_load(y_true, y_pred, metric_fn, n_iters):
    load_errors = []
    inds = np.arange(y_true.shape[0])
    for i in range(n_iters):
        inds_boot = resample(inds)
        y_true_boot = y_true[inds_boot]
        y_pred_boot = y_pred[inds_boot]
        load_errors.append(metric_fn(y_true_boot, y_pred_boot))
    return np.mean(load_errors, axis=0), np.std(load_errors, axis=0)


def mean_estimation_absolute_percentage_error(y_true, y_pred, eps=1e-8, n_iters=100):   
    def metric_fn(y_true, y_pred, eps=eps):
        return np.abs((y_true.mean(0)-y_pred.mean(0))/(eps+y_true.mean(0))).squeeze()*100
    return _eval_load(y_true, y_pred, metric_fn, n_iters)

def std_estimation_absolute_percentage_error(y_true, y_pred, eps=1e-8, n_iters=100):   
    def metric_fn(y_true, y_pred, eps=eps):
        return np.abs((y_true.std(0)-y_pred.std(0))/(eps+y_true.std(0))).squeeze()*100
    return _eval_load(y_true, y_pred, metric_fn, n_iters)

