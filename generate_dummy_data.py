import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def load_word_list(filepath, vocab_size=5000):
    """Loads the large word list from the specified file."""
    try:
        with open(filepath, 'r') as f:
            words = [line.strip() for line in f if len(line.strip()) > 2]
        print(f'Successfully loaded {len(words)} words from {filepath}.')

        if len(words) < vocab_size:
            print(f'Warning: The word list contains fewer words ({len(words)}) than the desired vocab size ({vocab_size}). Using all available words.')
            return words
        
        word_bank = random.sample(words, vocab_size)
        return word_bank
    except FileNotFoundError:
        print(f"Error: The word list file was not found at '{filepath}'.")
        print("Please download the 'english_words.txt' file and place it in the project's 'data/' directory.")
        return None

# --- NEW: Function to generate realistic, variable-length text ---
def generate_random_text(word_bank, min_words=3, max_words=15):
    """
    Generates a random sentence-like string with a variable number of words
    by sampling from the provided word bank.
    """
    if not word_bank:
        return "default placeholder text"
        
    num_words = random.randint(min_words, max_words)
    # Use random.choices for performance with large lists
    return " ".join(random.choices(word_bank, k=num_words))


def generate_dummy_data(num_rows=5000, output_path='data/Sample_Data.csv'):
    """
    Generates a synthetic dataset that mimics the structure of the original
    challenge data, including variable-length text fields.
    """
    word_bank = load_word_list('data/english_words.txt')
    if word_bank is None:
        return
    
    print(f"Generating {num_rows} rows of synthetic data...")

    # Sample categories for the categorical column
    categorical_samples = ['Type A', 'Type B', 'Type C', 'Type D', 'Type E']

    # --- Generate Data for Each Column using the new function ---
    data = {
        # Text Columns with variable length
        'Col1': [generate_random_text(word_bank) for _ in range(num_rows)],
        'Col2': [generate_random_text(word_bank) for _ in range(num_rows)],
        'Col4': [generate_random_text(word_bank) for _ in range(num_rows)],
        'Col6': [generate_random_text(word_bank) for _ in range(num_rows)],

        # Numerical Column
        'Col3': np.random.normal(loc=15000, scale=8000, size=num_rows).round(2),

        # Date Column
        'Col5': [(datetime.now() - timedelta(days=random.randint(0, 730))).strftime('%Y-%m-%d %H:%M:%S') for _ in range(num_rows)],

        # Categorical Column
        'Col7': [random.choice(categorical_samples) for _ in range(num_rows)],

        # Target Label (with severe imbalance)
        'ClassificationLabel': random.choices(
            population=['category0', 'category_0', 'category1', 'category2', 'category3', 'catgry_3', 'category4', 'category5'],
            weights=[0.85, 0.0158, 0.1236, 0.0046, 0.003, 0.0002, 0.0024, 0.0004],
            k=num_rows
        )
    }

    df = pd.DataFrame(data)

    # Introduce some missing values
    for col in df.columns:
        if col != 'ClassificationLabel':
            missing_indices = df.sample(frac=0.03).index
            df.loc[missing_indices, col] = np.nan

    # --- Save the DataFrame to a CSV File ---
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    df.to_csv(output_path, index=False)
    print(f"Synthetic data with variable text length successfully generated and saved to '{output_path}'")


if __name__ == "__main__":
    generate_dummy_data()