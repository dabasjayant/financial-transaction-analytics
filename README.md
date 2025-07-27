# Data Science Project - Financial Transaction Analytics

This repository contains a **personal data science project** developed as part of the **EY Data Science Take-Home Challenge (Summer 2025)**.  

The solution demonstrates a **professional DS approach to building a classification pipeline** for transaction-like data with **significant class imbalance, mixed data types, and unstructured text fields**.

## Project Overview

The goal was to design a **reliable and scalable framework** for classifying records into multiple categories, balancing accuracy with computational efficiency.  
Key considerations included:  
- Handling **imbalanced class distributions** (98.8% of records in two categories).  
- Processing **text, numeric, and temporal features** without domain-specific context.  
- Building a pipeline that could be **generalized for financial analytics use cases** such as **fraud detection, transaction monitoring, or customer segmentation**.

## Dataset Characteristics

- **Total Records:** 5,899  
- **Features:**  
  - `Col1, Col2, Col4, Col6`: Text fields (multi-token strings)  
  - `Col3`: Numeric  
  - `Col5`: Datetime (expanded to `year`, `month`, `day`)  
  - `Col7`: Categorical (low cardinality)  
- **Target:** `ClassificationLabel` (categorical string)  
- **Class Distribution:** Extreme imbalance; minority classes with as few as 1–5 samples.

## Data Engineering

- **Missing Data:** Removed 153 rows (2.6%) containing nulls, primarily from majority classes.  
- **Label Normalization:** Standardized inconsistent label formats (e.g., `Category2` → `Category_2`) using `fuzzywuzzy`.  
- **Feature Processing:**  
  - TF-IDF vectorization for text fields (word-level importance).  
  - Date decomposition into year, month, day components.  
  - Label encoding for categorical features and target.  
- **Class Imbalance Strategies:**  
  - Compared **oversampling/undersampling** (equal class counts) with a **class weighting approach**.  
  - Adopted **class weights with moderate oversampling** to reduce overfitting while maintaining recall for minority classes.

## Modeling Approach

Implemented and tuned three supervised models using **GridSearchCV**:  
- **Random Forest** – Chosen for final deployment (balanced accuracy, training time, and robustness).  
- **Gradient Boosting** – High accuracy but ~30 minutes training per run.  
- **Support Vector Machine (SVM)** – Improved with class weights but underperformed compared to ensembles.

### Metrics
- **Accuracy** – Overall classification rate.  
- **Weighted F1-Score** – Accounts for class imbalance.  
- **Cross-Entropy Loss** – Evaluates probability calibration.

## Performance Summary

| Model               | Accuracy | Weighted F1 | Log Loss | Training Time (s) |
|---------------------|----------|-------------|----------|--------------------|
| SVM (weighted)      | 0.89     | 0.84        | –        | 84.3               |
| Gradient Boost      | 0.96     | 0.96        | 0.15     | 1,769.6            |
| **Random Forest**   | **0.97** | **0.97**    | **0.08** | 19.5               |

**Final Model:** Hypertuned RandomForest Classifier (best trade-off of accuracy, stability, and computational cost).  
**Notable Insights:**  
- TF-IDF features from `Col1` and `Col6` were primary drivers of model performance.  
- Numeric feature `Col3` and temporal features added secondary predictive value.  
- Minimal seasonal effects, with some class-specific month patterns (e.g., Label 4 only in October).

## Key Takeaways

1. **Text-driven features are critical** – Word-level patterns strongly correlated with labels.  
2. **Class weights outperform oversampling** – Reduced overfitting while improving recall for underrepresented categories.  
3. **Dimensionality reduction (PCA/LSA)** could further optimize the high-dimensional TF-IDF feature space for scalability.

## Future Enhancements

- Add **PCA or Latent Semantic Analysis** for dimensionality reduction.  
- Integrate **XGBoost/LightGBM** for faster gradient boosting alternatives.  
- Package as a **reusable Python module** for enterprise data science workflows in **banking, fraud detection, and financial risk analytics**.

## Running the Project

```bash
git clone https://github.com/<your-username>/transaction-data-science-pipeline.git
cd transaction-data-science-pipeline
pip install -r requirements.txt
jupyter notebook JayantDabas_project_EY.ipynb
