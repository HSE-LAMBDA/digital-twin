import numpy as np
from sklearn import metrics
import joblib
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def mmd_rbf(X, Y, gamma=None, n_iters=1, n_jobs=6, standardize=True):
    """ Bootstrap MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))"""
    
    if standardize:
        scaler =  StandardScaler()
        X = scaler.fit_transform(X)
        Y = scaler.transform(Y)
    if gamma is None:
        agg_matrix = np.concatenate((X, Y), axis=0)
        distances = metrics.pairwise_distances(agg_matrix)
        median_distance = np.median(distances)
        gamma = 1.0 / (2 * median_distance**2)
        
    
    def _mmd_rbf(args):
        assert len(args) == 2, "Expected 2 arguments, got {}".format(len(args))
        _X, _Y = args
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.rbf_kernel(_X, _X, gamma)
        YY = metrics.pairwise.rbf_kernel(_Y, _Y, gamma)
        XY = metrics.pairwise.rbf_kernel(_X, _Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()
    mmds = joblib.Parallel(n_jobs=n_jobs, verbose=0)(joblib.delayed(_mmd_rbf)(resample(X, Y))  for _ in range(n_iters))
    return np.mean(mmds), np.std(mmds)
    