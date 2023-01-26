import numpy as np
from scipy.linalg import sqrtm
from sklearn.utils import resample


def bootstrap_frdist(y_true, y_pred, n_iters=100):
    frd = []
    
    inds = np.arange(len(p))
    
    for i in range(n_iters):
        inds_boot = resample(inds)
        
        y_true_boot = y_true[inds_boot]
        y_pred_boot = y_pred[inds_boot]
        
        y_true_mean, y_true_cov = y_true_boot.mean(axis=0), np.cov(y_true_boot, rowvar=False)
        y_pred_mean, y_pred_cov = y_pred_boot.mean(axis=0), np.cov(y_pred_boot, rowvar=False)
        
        diff = np.sum((y_true_mean - y_pred_mean)**2.0)
        covmean, _ = sqrtm(y_true_cov.dot(y_pred_cov), disp=False)
        
        if np.iscomplexobj(covmean): covmean = covmean.real
        tr_covmean = np.trace(covmean)
            
        ifrd = diff + np.trace(y_true_cov) + np.trace(y_pred_cov) - 2 * tr_covmean
        frd.append(ifrd)
        
    frd = np.array(frd)
    return frd.mean(axis=0), frd.std(axis=0)
