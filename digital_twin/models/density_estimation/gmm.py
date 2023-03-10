import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
import logging

logger = logging.getLogger(__name__)

class GMM:
    """Gaussian Mixture Model"""

    def __init__(
        self,
        n_components=2,
        covariance_type="full",
        warm_start=True,
        n_init=5,
        init_params="kmeans",
        tol=1e-3,
        use_elliptic_envelope=False,
        elliptic_envelope_contamination=0.05,
        random_state=666,
    ):
        self.use_elliptic_envelope = use_elliptic_envelope
        self.elliptic_envelope_contamination = elliptic_envelope_contamination
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            warm_start=warm_start,
            init_params=init_params,
            tol=tol,
            n_init=n_init,
            random_state=random_state,
        )
    
    def fit(self, X):
        """ Fit the GMM. """
        if self.use_elliptic_envelope:
            try:
                self.elliptic_envelope = EllipticEnvelope(
                    contamination=self.elliptic_envelope_contamination, random_state=self.random_state
                )
                mask = self.elliptic_envelope.fit_predict(X)
                X = X[mask == 1]
            except ValueError:
                logger.error("Not enough data to fit EllipticEnvelope")
                logger.error(X.describe())
        self.model.fit(X)
        logger.info(f"Model converged: {self.model.converged_}")
    
    def init_from_params(self, weights, means, precisions_cholesky):
        self.model.weights_ = weights 
        self.model.means_ = means
        self.model.precisions_cholesky_ = self.triu_to_full_precisions_cholesky(precisions_cholesky)
        # need to be more generic
        covariances = np.linalg.inv(self.model.precisions_cholesky_ @ np.transpose(self.model.precisions_cholesky_, (0, 2, 1)))
        self.model.covariances_ = covariances
        logger.info(f"Model parameters initialized from params")
        
    def sample(self, n_samples: int = 1000):
        """ Sample from the GMM. """
        samples, _ = self.model.sample(n_samples)
        return np.trunc(np.expm1(samples))

    @property
    def weights(self):
        # (2,)
        return self.model.weights_
    
    @property
    def means(self):
        # (2, 2)
        return self.model.means_
    
    @property
    def covariances(self):
        # (2, 2, 2)
        return self.model.covariances_
    
    @property
    def precisions_cholesky(self):
        rows, cols = np.triu_indices(self.model.precisions_cholesky_.shape[1])
        return self.model.precisions_cholesky_[:, rows, cols]
    
    def triu_to_full_precisions_cholesky(self, upper_triangular_elements):
        # https://oeis.org/A003056
        n = np.floor((np.sqrt(1+8*upper_triangular_elements.shape[1])-1)/2).astype(int)
        reconstructed_matrices = np.zeros((upper_triangular_elements.shape[0], n, n))
        row, col = np.triu_indices(n)
        reconstructed_matrices[:, row, col] = upper_triangular_elements
        # self.model.precisions_cholesky_ = reconstructed_matrices
        return reconstructed_matrices
