# 🤖 Customer Churn Prediction: Machine Learning Classification

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Overview

This project builds machine learning models to predict customer churn for a telecommunications company. Using the insights from our EDA (Project 1), we train, compare, and optimize multiple classification algorithms.

**Business Goal:** Identify customers likely to churn so the company can take proactive retention actions.

## 🎯 Key Results

| Model | ROC-AUC | F1-Score | Recall |
|-------|---------|----------|--------|
| **Tuned XGBoost** | **0.79** | **0.54** | **0.68** |
| Random Forest | 0.73 | 0.50 | 0.62 |
| Logistic Regression | 0.73 | 0.51 | 0.64 |
| SVM | 0.71 | 0.49 | 0.60 |

## 📁 Project Structure

```
02-classification-ml/
├── README.md
├── churn_classification_model.ipynb    # Main modeling notebook
├── data/
│   └── telecom_churn_cleaned.csv       # Cleaned dataset from EDA
├── models/
│   ├── xgboost_churn_model.pkl         # Trained model
│   ├── scaler.pkl                      # Feature scaler
│   └── feature_columns.pkl             # Feature column names
└── images/                             # Generated visualizations
    ├── 01_model_comparison.png
    ├── 02_roc_curves.png
    ├── 03_confusion_matrices.png
    ├── 04_feature_importance.png
    └── ...
```

## 📊 Analysis Sections

1. **Data Loading & Preprocessing** - Feature engineering, encoding, scaling
2. **Train-Test Split** - Stratified 80/20 split
3. **Class Imbalance Handling** - SMOTE oversampling
4. **Model Training** - 4 algorithms compared
5. **Model Evaluation** - ROC-AUC, F1, Precision, Recall
6. **Hyperparameter Tuning** - GridSearchCV for XGBoost
7. **Feature Importance** - Top predictive features
8. **Cross-Validation** - 5-fold CV for robustness
9. **Business Impact Analysis** - Financial projections
10. **Model Persistence** - Saved for deployment

## 🔬 Models Evaluated

- **Logistic Regression** - Baseline interpretable model
- **Random Forest** - Ensemble of decision trees
- **XGBoost** - Gradient boosting (best performer)
- **Support Vector Machine** - Kernel-based classification

## 📈 Key Visualizations

### ROC Curves Comparison
![ROC Curves](images/02_roc_curves.png)

### Feature Importance (XGBoost)
![Feature Importance](images/04_feature_importance.png)

### Business Impact
![Business Impact](images/07_business_impact.png)

## 🔍 Top Predictive Features

1. **Contract_Two Year** (0.104) - Long contracts = low churn
2. **Contract_One Year** (0.097) - Annual contracts reduce churn
3. **PaymentMethod_Electronic Check** (0.092) - High churn risk
4. **Referrals** (0.059) - Loyal customers refer others
5. **InternetService_Fiber Optic** (0.054) - Competition factor
6. **Tenure_Months** (0.042) - Longer tenure = retention

## 💰 Business Impact

Based on test set predictions:
- **Customers identified for intervention:** ~230
- **Estimated customers saved (30% success rate):** ~50
- **Projected revenue protected:** $78,000+
- **Net benefit after intervention costs:** $73,000+

## 🛠️ Technologies Used

- **Python 3.12**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - ML algorithms, preprocessing, evaluation
- **XGBoost** - Gradient boosting
- **Imbalanced-learn** - SMOTE for class imbalance
- **Matplotlib, Seaborn** - Visualizations

## 🚀 How to Use

### Run the Analysis
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
jupyter notebook churn_classification_model.ipynb
```

### Load the Trained Model
```python
import pickle

# Load model
with open('models/xgboost_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
# X_new = preprocess_new_data(...)  # Your new data
# predictions = model.predict(X_new)
```

## 📌 Business Recommendations

1. **Deploy Real-Time Scoring** - Integrate model into CRM
2. **Prioritize High-Risk** - Focus on customers with P(churn) > 0.6
3. **Monthly Batch Scoring** - Identify new at-risk customers
4. **A/B Test Interventions** - Validate retention campaigns
5. **Monitor & Retrain** - Update model quarterly

## 🔗 Related Projects

- **[Project 1: EDA](../01-exploratory-data-analysis/)** - Exploratory analysis that informed this model
- **[Project 3: Regression](../03-regression-ml/)** - Price prediction (coming next)

## 👤 Author

[Your Name]  
Data Analyst | [Your LinkedIn] | [Your GitHub]

---

*This project is part of my Data Analysis Portfolio. See the [main repository](../) for more projects.*
