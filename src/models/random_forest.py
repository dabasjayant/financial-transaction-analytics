from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV

class RandomForest:
    def __init__(self, random_state=0):
        self.name = 'random_forest_classifier'
        self.model = RandomForestClassifier(random_state=random_state)

    def train(self, X, y, pipeline: ImbPipeline, cv=5, scoring='accuracy'):
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 5, 7],
        }
        return GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    
    def get_model(self):
        return self.model

    def get_name(self):
        return self.name