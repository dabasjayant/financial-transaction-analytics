from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

class XGBoost:
    def __init__(self, random_state=0):
        self.name = 'xg_boost_classifier'
        self.model = XGBClassifier(random_state=random_state)
    
    def get_model(self):
        return self.model

    def get_name(self):
        return self.name