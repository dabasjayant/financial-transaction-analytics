import numpy as np

from config.config import Config
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.classifier import Classifier

from sklearn.model_selection import train_test_split

def main():
    config = Config()
    
    # --- Data Preprocessing ---
    data_preprocessor = DataPreprocessor(config)
    data = data_preprocessor.load_data()
    data_clean = data_preprocessor.clean_data(data)
    data_encoded = data_preprocessor.encode_labels(data_clean)

    # --- Feature Engineering ---
    feature_engineer = FeatureEngineer(config)
    date_column = config.DATE_COLUMN[0]
    # Creates features based on specific date
    data_with_features = feature_engineer.create_event_features(data_encoded, date_column)
    # Creates 'day', 'month', 'year' and drop Col5
    data_with_features = feature_engineer.create_date_features(data_with_features, date_column)
    # Create cyclical features for month and day
    data_with_features = feature_engineer.create_cyclical_features(data_with_features, 'month')
    data_with_features = feature_engineer.create_cyclical_features(data_with_features, 'day')
    # Transform text to vectors using TF-IDF
    text_features_sparse = feature_engineer.create_text_features(data_with_features)

    # --- PCA on Text Features ---
    print(f'Original shape of text features: {text_features_sparse.shape}')
    text_features_pca = feature_engineer.apply_pca(text_features_sparse, 0.95)
    print(f'Shape of text features after PCA: {text_features_pca.shape}')

    # --- Combine All Features for Modeling ---
    numerical_cols = config.NUMERICAL_COLUMNS + ['year', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'day_of_week', 'is_weekend', 'days_since_start']
    X_numerical = data_with_features[numerical_cols].values
    X_categorical = data_with_features[config.CATEGORICAL_COLUMNS].values
    # Combine all feature sets into the final design matrix X
    X = np.hstack([X_numerical, X_categorical, text_features_pca])
    y = data_with_features[config.TARGET_COLUMN]
    print(f'Final shape of combined feature matrix X: {X.shape}')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0, stratify=y)

    # --- Model Training ---
    model_name = ['random_forest']
    for i, name in enumerate(model_name):
        print(f'[{i}/{len(model_name)}] Using {name}')
        model = Classifier(config, name)
        print('Training...')
        model.train(X_train, y_train)
        print(f'Best model parameters: {model.get_best_model()}')
        print('Testing...')
        model.test(X_test, y_test)

    print('Complete!')

if __name__ == '__main__':
    main()