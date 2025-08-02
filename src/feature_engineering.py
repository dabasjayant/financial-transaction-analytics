import numpy as np

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def create_date_features(self, data, column_name):
        data['year'] = data[column_name].dt.year
        data['month'] = data[column_name].dt.month
        data['day'] = data[column_name].dt.day
        return data.drop(columns=[column_name])

    def create_text_features(self, data):
        text_data = data[self.config.TEXT_COLUMNS].apply(lambda x: ' '.join(x), axis=1)
        vectorizer = TfidfVectorizer()
        text_features = vectorizer.fit_transform(text_data)
        return text_features
    
    def create_cyclical_features(self, data, column_name):
        """
        Creates sine and cosine features for cyclical data like month or day.
        """
        # Get the maximum value for the cycle (e.g., 12 for month, 31 for day)
        max_val = data[column_name].max()
        
        data[column_name + '_sin'] = np.sin(2 * np.pi * data[column_name] / max_val)
        data[column_name + '_cos'] = np.cos(2 * np.pi * data[column_name] / max_val)
        
        return data.drop(columns=[column_name])
    
    def create_event_features(self, data, column_name):
        """
        Creates features based on specific date events.
        """
        data['day_of_week'] = data[column_name].dt.dayofweek # Monday=0, Sunday=6
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int) # boolean
        
        # Calculates the number of days from the earliest date in the dataset
        # It helps in capturing linear trends.
        min_date = data[column_name].min()
        data['days_since_start'] = (data[column_name] - min_date).dt.days
        
        return data
    
    def apply_pca(self, data, variance=0.9):
        # Scale features before PCA (important)
        scaler = StandardScaler(with_mean=False)
        text_features_scaled = scaler.fit_transform(data)

        # Apply PCA to reduce dimensionality, retaining 95% of the variance
        pca = PCA(n_components=variance)
        return pca.fit_transform(text_features_scaled.toarray()) # .toarray() is needed for PCA