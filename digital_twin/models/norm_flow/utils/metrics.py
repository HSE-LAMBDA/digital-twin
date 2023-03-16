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


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score


def discrepancy_score(observations, forecasts, model='QDA', n_iters=1):
    """
    Parameters:
    -----------
    observations : numpy.ndarray, shape=(n_samples, n_features)
        True values.
        Example: [[1, 2], [3, 4], [4, 5], ...]
    forecasts : numpy.ndarray, shape=(n_samples, n_features)
        Predicted values.
        Example: [[1, 2], [3, 4], [4, 5], ...]
    model : sklearn binary classifier
        Possible values: RF, DT, LR, QDA, GBDT
    n_iters : int
        Number of iteration per one forecast.
        
    Returns:
    --------
    mean : float
        Mean value of discrepancy score.
    std : float
        Standard deviation of the mean discrepancy score.
    
    """
    
    
    scores = []

    X0 = observations
    y0 = np.zeros(len(observations))
    
    X1 = forecasts
    y1 = np.ones(len(forecasts))
    
    X = np.concatenate((X0, X1), axis=0)
    y = np.concatenate((y0, y1), axis=0)
        
    for it in range(n_iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)
        if model == 'RF':
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=None)
        elif model == 'GDBT':
            clf = GradientBoostingClassifier(max_depth=6, subsample=0.7)
        elif model == 'DT':
            clf = DecisionTreeClassifier(max_depth=10)
        elif model == 'LR':
            clf = LogisticRegression()
        elif model == 'QDA':
            clf = QuadraticDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict_proba(X_test)[:, 1]
        auc = 2 * roc_auc_score(y_test, y_pred_test) - 1
        scores.append(auc)

    scores = np.array(scores)
    mean = scores.mean()
    std  = scores.std() / np.sqrt(len(scores))
    
    return mean, std
