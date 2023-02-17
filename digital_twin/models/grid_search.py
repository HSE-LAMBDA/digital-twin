from sklearn.model_selection import GridSearchCV

class GridSearchCV(GridSearchCV):
    def predict(self, X, **params):
        return self.best_estimator_.predict(X, **params)
