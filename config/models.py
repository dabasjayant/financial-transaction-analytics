from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class RandomForest:
    def __init__(self, random_state=0):
        self.name = 'random_forest_classifier'
        self.model = RandomForestClassifier(random_state=random_state)
        self.param_grid = {
            'n_estimators': [100, 150, 200],
            'criterion': ['gini', 'log_loss'],
            'max_depth': [None, 5, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'bootstrap': [True]
        }
    
    def get_model(self):
        return self.model

    def get_name(self):
        return self.name

    def get_param_grid(self):
        return self.param_grid

class XGBoost:
    def __init__(self, random_state=0):
        self.name = 'xg_boost_classifier'
        self.model = XGBClassifier(random_state=random_state)
        self.param_grid = {}
    
    def get_model(self):
        return self.model

    def get_name(self):
        return self.name

    def get_param_grid(self):
        return self.param_grid