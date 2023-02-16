from catboost import CatBoostRegressor

class Regressor:
    def __init__(self,regressor_params, regressor_type="catboost"):
        self.regressor_params = regressor_params
        self.regressor_type = regressor_type
        
    def fit(self, X, y, *args, **kwargs):
        if self.regressor_type == "catboost":
            self.model = CatBoostRegressor(**self.regressor_params, cat_features=[col for col in X if isinstance(X[col].iloc[0], str)])
            return self.model.fit(X, y, *args, **kwargs)
        else:
            raise NotImplementedError(f"Regressor type {self.regressor_type} not implemented.")
        
    def predict(self, X):
        return self.model.predict(X)

    def load_model(self, path):
        self.model = CatBoostRegressor()
        self.model.load_model(path)
        
    def save_model(self, path):
        self.model.save_model(path, format="cbm")