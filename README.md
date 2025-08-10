# End-to-End Classification Pipeline for High-Dimensional, Imbalanced Data

This repository contains the source code for a robust, end-to-end multi-class classification pipeline built entirely in Python. The project demonstrates a comprehensive, professional approach to a realistic business problem involving severely imbalanced, high-dimensional, and mixed-type data. The entire pipeline is built with a modular, scalable, and production-ready code structure, moving beyond a standard Jupyter Notebook to showcase enterprise-level software engineering practices.

**Note:** The original dataset is proprietary and is not included. However, the repository includes a synthetic data generator that allows the entire pipeline to be executed for demonstration and validation of the methodology.

---

## Technical Deep Dive & Skills Demonstrated

This project showcases a range of advanced data science skills and best practices critical for a professional data scientist role.

### 1. Advanced Feature Engineering & Preprocessing
-   **Holistic NLP Pipeline**: To capture the complete semantic context for each record, all text columns were strategically combined into a single document before applying a TF-IDF vectorizer. This approach creates one unified, high-dimensional feature set representing the full textual signature of each transaction, a robust method for many classification tasks.
-   **Sophisticated Datetime Feature Creation**: Moved beyond simple day/month/year splits to engineer more insightful features:
    -   **Cyclical Features**: Transformed month and day-of-month into `sin`/`cos` components to represent their cyclical nature mathematically, allowing the model to understand that December is as close to January as March is to April.
    -   **Event-Based & Trend Features**: Created boolean (`is_weekend`) and linear (`days_since_start`) features to capture distinct weekly patterns and long-term trends in the data.
-   **Robust Data Cleaning**: Implemented systematic cleaning for mixed-type data, including outlier capping for numerical columns and robust imputation strategies to handle missing values without discarding valuable data rows.

### 2. Dimensionality Reduction & Data Synthesis
-   **Principal Component Analysis (PCA)**: Effectively managed the high-dimensional feature space created by the TF-IDF vectorizer. PCA was applied to the combined text features to reduce dimensionality, remove noise, and improve model training efficiency and generalization.
-   **Expert Handling of Severe Class Imbalance**: Addressed the core challenge of a severely imbalanced dataset by integrating **SMOTE (Synthetic Minority Over-sampling TEchnique)** into the modeling pipeline. This was implemented correctly within an `imblearn` pipeline to prevent data leakage during cross-validation—a critical step that ensures the model's performance metrics are valid and not artificially inflated. The `k_neighbors` parameter was carefully managed to handle extremely rare classes with very few samples.

### 3. Efficient Model Tuning & Rigorous Evaluation
-   **Intelligent Hyperparameter Search**: Utilized **`RandomizedSearchCV`** instead of an exhaustive `GridSearchCV`. This demonstrates a practical understanding of computational trade-offs, allowing for efficient yet effective exploration of a large hyperparameter space to find a high-performing model configuration.
-   **Metrics-Driven Evaluation**: Focused on appropriate evaluation metrics for an imbalanced classification problem. The primary success metric was the **Weighted F1-Score**, supplemented by a detailed **Classification Report** and **Confusion Matrix** analysis to understand the model's performance on a per-class basis, rather than relying on misleading overall accuracy.

### 4. Enterprise-Level Python & Software Engineering
-   **Modular & Scalable Architecture**: The project was architected with a professional, modular structure that separates concerns into distinct components (`config`, `src`, `main.py`). This promotes code that is clean, readable, maintainable, and easily scalable for production environments.
-   **Reproducibility & Dependency Management**: The project is fully reproducible using a `requirements.txt` file and includes a runbook for environment setup, demonstrating a commitment to professional software development practices.

---

## Runbook: Setup & Execution

Follow these instructions to set up the project environment and run the complete pipeline.

### 1. Prerequisites
-   Python 3.8+
-   [uv](https://github.com/astral-sh/uv) (A fast, modern Python package installer and resolver)

### 2. Environment Setup
First, clone the repository to your local machine:
```bash
git clone <your-github-repository-url>
cd financial-transaction-analytics
```

Next, create and activate a virtual environment using `uv`. This is faster and more efficient than traditional `venv` and `pip`.

```bash
# Create a new virtual environment named .venv
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (CMD):
.venv\Scripts\activate
```

With the environment activated, install the required Python packages using `uv pip`:

```bash
uv pip install -r requirements.txt
```

### 3. Generate Synthetic Data
The original dataset for this project is proprietary. A script is provided to generate a structurally-identical synthetic dataset, which allows the pipeline to be run end-to-end for demonstration.

Execute the following command from the root directory:

```bash
python generate_dummy_data.py
```

This will create a `Sample_Data.csv` file inside the `/data` directory.

### 4. Run the Pipeline
To execute the entire data processing, feature engineering, and model training pipeline, run the `main.py` script from the root directory:

```bash
python main.py
```

The script will output the progress of each stage of the pipeline. Upon completion, it will print the model evaluation results (including the final F1-score and classification report) and display a confusion matrix plot. The final trained model object will be saved as `random_forest_model.pkl` in a newly created `/models` directory.

## Test Results

```bash
> python main.py
[nltk_data] Downloading package wordnet to
[nltk_data]     ~/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package stopwords to
[nltk_data]     ~/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Original shape of text features: (5746, 6826)
Shape of text features after PCA: (5746, 4393)
Final shape of combined feature matrix X: (5746, 4403)
[1/1] Using random_forest
Training...
Fitting 5 folds for each of 50 candidates, totalling 250 fits

Method runtime: 2540.372555 seconds
Best model parameters: RandomForestClassifier(criterion='log_loss', min_samples_split=5,
                       n_estimators=150, random_state=0)
Testing...
Accuracy: 0.9977011494252873
F1 Score: 0.9977011360367974
Log Loss: 0.03417957842435477
Classification report: 
              precision    recall  f1-score   support

           0       1.00      0.99      0.99       508
           1       0.99      1.00      0.99       507
           2       1.00      1.00      1.00       507
           3       1.00      1.00      1.00       508
           4       1.00      1.00      1.00       508
           5       1.00      1.00      1.00       507

    accuracy                           1.00      3045
   macro avg       1.00      1.00      1.00      3045
weighted avg       1.00      1.00      1.00      3045

Complete!
```

## TODO

1. Upload python notebook for data exploration and playground.
2. Migrate visualization class from python notebook to native python scripts