import pickle
import os

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss

from config.models import RandomForest, XGBoost

class Classifier:
    def __init__(self, config, model_name: str):
        self.config = config
        self.random_state = 0
        self.classifier = RandomForest(self.random_state) 
        if model_name == 'xgboost':
            self.classifier = XGBoost(self.random_state)

        model = self.classifier.get_model()
        param_grid = self.classifier.get_param_grid()
        # self.search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
        self.search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=self.random_state)

    def train(self, X, y):
        self.search.fit(X, y)

    def test(self, X, y):
        model = self.get_best_model()
        y_pred = model.predict(X)
        
        print(f'Accuracy: {accuracy_score(y, y_pred)}')
        print(f'F1 Score: {f1_score(y, y_pred, average="weighted")}')
        print(f'Log Loss: {log_loss(y, model.predict_proba(X))}')
        print(f'Classification report: \n{classification_report(y, y_pred)}')

    def save_best_model(self):
        model = self.get_best_model()
        base_dir = self.config.MODEL_PATH
        try:
            os.makedirs(base_dir, exist_ok=True)
        except OSError as e:
            print(f'Error creating saved_models directory \'{base_dir}\': {e}')
        with open(f'{base_dir}/{self.classifier.get_name()}.pkl', 'wb') as f:
            pickle.dump(model, f)

    def get_best_model(self):
        return self.search.best_estimator_