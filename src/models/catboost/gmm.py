import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
import logging

logger = logging.getLogger(__name__)

class GMM:
    """Gaussian Mixture Model"""

    def __init__(
        self,
        X,
        n_components,
        covariance_type,
        warm_start=True,
        n_init=5,
        init_params="kmeans",
        tol=1e-3,
        use_elliptic_envelope=False,
        elliptic_envelope_contamination=0.05,
        random_state=666,
    ):
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            warm_start=warm_start,
            init_params=init_params,
            tol=tol,
            n_init=n_init,
            random_state=random_state,
        )

        self.use_elliptic_envelope = use_elliptic_envelope
        if use_elliptic_envelope:
            try:
                self.elliptic_envelope = EllipticEnvelope(
                    contamination=elliptic_envelope_contamination, random_state=random_state
                )
                mask = self.elliptic_envelope.fit_predict(X)
                X = X[mask == 1]
            except ValueError:
                logger.error("Not enough data to fit EllipticEnvelope")
                logger.error(X.describe())
                
        self.model.fit(X)
        
    
    @property
    def weights(self):
        return self.model.weights_
    
    @property
    def means(self):
        return self.model.means_
    
    @property
    def covariances(self):
        return self.model.covariances_
    
    @property
    def precisions_cholesky(self):
        return self.model.precisions_cholesky_
    
    @weights.setter
    def weights(self, weights):
        self.model.weights_ = weights
    
    @means.setter
    def means(self, means):
        self.model.means_ = means
    
    @covariances.setter
    def covariances(self, covariances):
        self.model.covariances_ = covariances
        
    @precisions_cholesky.setter
    def precisions_cholesky(self, precisions_cholesky):
        self.model.precisions_cholesky_ = precisions_cholesky