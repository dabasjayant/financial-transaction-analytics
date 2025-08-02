import pickle
import os

from src.models.random_forest import RandomForest
from src.models.xg_boost import XGBoost

from datetime import datetime
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, log_loss

class Model:
    def __init__(self, config, model: str):
        self.config = config
        self.random_state = 0
        self.model = RandomForest(self.random_state) if model == 'random_forest' else XGBoost(self.random_state)        

    def train_test_split(self, X, y, size=0.9):
        return train_test_split(X, y, test_size=size, random_state=self.random_state, stratify=y)

    def train(self, X, y):
        smote = SMOTE(random_state=self.random_state, k_neighbors=1)
        pipeline = ImbPipeline([
            ('smote', smote),
            ('classifier', self.model.get_model())
        ])
        self.grid_search = self.model.train(X, y, pipeline, scoring='f1_weighted')

    def test(self, X, y):
        model = self.get_best_model()
        y_pred = model.predict(X, y)
        
        print(f'Accuracy: {accuracy_score(y, y_pred)}')
        print(f'F1 Score: {f1_score(y, y_pred, average='weighted')}')
        print(f'Log Loss: {log_loss(y, model.predict_proba(X))}')

    def save_best_model(self):
        model = self.get_best_model()
        base_dir = self.config.MODEL_PATH + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        try:
            os.makedirs(base_dir, exist_ok=True)
        except OSError as e:
            print(f'Error creating saved_models directory \'{base_dir}\': {e}')
        with open(f'{base_dir}/{self.model.get_name()}.pkl', 'wb') as f:
            pickle.dump(model, f)

    def get_best_model(self):
        return self.grid_search.best_estimator_