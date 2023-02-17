from sklearn.neighbors import KNeighborsRegressor

class KNN(KNeighborsRegressor):
    def predict(self, X, n_samples=1):
        if n_samples==1:
            return super().predict(X)
        else:
            neigh_inds = super().kneighbors(X, n_neighbors=n_samples, return_distance=False)
            y = self._y
            if y.ndim == 1:
                y = y.reshape((-1, 1))
            return y[neigh_inds]
