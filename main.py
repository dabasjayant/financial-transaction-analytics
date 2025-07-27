from src.data_preprocessing import DataPreprocessor
from config.config import Config

def main():
    config = Config()
    
    # Data Preprocessing
    data_preprocessor = DataPreprocessor(config)
    data = data_preprocessor.load_data()
    data_clean = data_preprocessor.clean_data(data)
    data_encoded = data_preprocessor.encode_labels(data_clean)

if __name__ == "__main__":
    main()