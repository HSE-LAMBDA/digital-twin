from sklearn.pipeline import Pipeline

class Pipeline(Pipeline):
    def predict(self, X, **predict_params):
        """Applies transforms to the data, and the predict method of the
        final estimator. Valid only if the final estimator implements
        predict."""
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        print(self.steps[-1][-1])
        return self.steps[-1][-1].predict(Xt, **predict_params)
