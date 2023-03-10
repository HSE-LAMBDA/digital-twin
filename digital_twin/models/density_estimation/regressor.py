from catboost import CatBoostRegressor
import logging 

logger = logging.getLogger(__name__)

class Regressor:
    def __init__(self,regressor_params, regressor_type="catboost"):
        self.regressor_params = regressor_params
        self.regressor_type = regressor_type
        
    def fit(self, X, y, grid_search_params=None, *args, **kwargs):
        if self.regressor_type == "catboost":
            self.model = CatBoostRegressor(**self.regressor_params, cat_features=[col for col in X if isinstance(X[col].iloc[0], str)])
            if grid_search_params is not None:
                self.model.set_params(use_best_model=False)
                best_params = self.model.grid_search(grid_search_params, X, y)
                logger.info("Best params: {}".format(best_params))
                return self.model
            else:
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