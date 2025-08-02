import nltk
import pandas as pd
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        nltk.download('wordnet')
        nltk.download('stopwords')

    def load_data(self):
        return pd.read_csv(self.config.DATA_PATH)

    def clean_data(self, data):
        data_clean = data.dropna().copy()
        for col in self.config.NUMERICAL_COLUMNS:
            data_clean[col] = pd.to_numeric(
                data_clean[col].apply(self._clean_num), errors='coerce'
            )
            data_clean = self.handle_outliers(
                data_clean, self.config.NUMERICAL_COLUMNS[0]
            )
        for col in self.config.DATE_COLUMN:
            data_clean[col] = pd.to_datetime(data_clean[col], errors='coerce')
        for col in self.config.TEXT_COLUMNS:
            data_clean[col] = data_clean[col].apply(self._clean_text)
        data_clean[self.config.TARGET_COLUMN] = data_clean[
            self.config.TARGET_COLUMN
        ].apply(self._clean_labels)
        return data_clean

    def _clean_num(self, text):
        """
        A function to clean numeric data.
        """
        # Ensure text is a string
        text = str(text).lower()
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        return text.replace(r'[^\\d.]', '', True)

    def _clean_text(self, text):
        """
        A function to clean text data.
        """
        # Ensure text is a string
        text = str(text).lower()
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [w for w in words if not w in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
        return ' '.join(words)

    def _clean_labels(self, label):
        digits = re.findall(r'\d+', label)
        if digits:
            extracted_digit = digits[0]
            return f'category{extracted_digit}'
        else:
            return label

    def encode_labels(self, data):
        encoder = LabelEncoder()
        for column in self.config.CATEGORICAL_COLUMNS + [self.config.TARGET_COLUMN]:
            data[column] = encoder.fit_transform(data[column])
        return data

    def handle_outliers(self, data, column):
        """
        Detects outliers using the IQR method and caps them.
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap the outliers
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
        return data
