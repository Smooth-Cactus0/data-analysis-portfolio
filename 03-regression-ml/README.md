# 🏠 House Price Prediction: Advanced Regression Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Overview

This project builds and compares multiple regression models to predict residential property prices. The analysis demonstrates advanced techniques including regularization (Ridge, Lasso, ElasticNet), ensemble methods (Random Forest, Gradient Boosting), and comprehensive model evaluation.

**Business Goal:** Develop an accurate price prediction model to help real estate agents, buyers, and sellers make data-driven decisions.

## 🎯 Key Results

| Model | R² Score | RMSE | MAE | MAPE |
|-------|----------|------|-----|------|
| **Tuned XGBoost** | **0.937** | **$39,892** | **$27,543** | **8.4%** |
| Gradient Boosting | 0.936 | $40,260 | $27,987 | 8.6% |
| Linear Regression | 0.927 | $42,756 | $31,253 | 10.0% |
| Ridge (L2) | 0.927 | $42,735 | $31,198 | 9.9% |
| Lasso (L1) | 0.927 | $42,756 | $31,247 | 9.9% |
| Random Forest | 0.894 | $51,519 | $35,876 | 11.4% |

**The model can predict house prices within ~8.4% error on average!**

## 📁 Project Structure

```
03-regression-ml/
├── README.md
├── house_price_regression.ipynb    # Main analysis notebook
├── data/
│   └── housing_prices.csv          # Housing dataset (3,000 properties)
├── models/
│   ├── xgboost_house_price_model.pkl   # Trained model
│   ├── scaler.pkl                       # Feature scaler
│   ├── feature_columns.pkl              # Feature names
│   └── feature_importance.csv           # Feature importance scores
└── images/                              # 10 visualizations
    ├── 01_price_distribution.png
    ├── 05_model_comparison.png
    ├── 06_residual_analysis.png
    ├── 08_feature_importance.png
    └── ...
```

## 📊 Analysis Sections

1. **Data Loading & EDA** - Price distribution, correlations, neighborhood analysis
2. **Data Cleaning** - Missing values, outlier detection and treatment
3. **Feature Engineering** - Created 8 new predictive features
4. **Model Training** - 7 regression algorithms compared
5. **Model Evaluation** - R², RMSE, MAE, MAPE metrics
6. **Residual Analysis** - Checking model assumptions
7. **Learning Curves** - Bias-variance tradeoff analysis
8. **Hyperparameter Tuning** - RandomizedSearchCV optimization
9. **Feature Importance** - Top predictive factors
10. **Cross-Validation** - 10-fold CV for robustness
11. **Business Application** - Price estimation example

## 🔬 Models Evaluated

| Category | Models |
|----------|--------|
| **Linear** | Linear Regression, Ridge (L2), Lasso (L1), ElasticNet |
| **Ensemble** | Random Forest, Gradient Boosting, XGBoost |

## 📈 Key Visualizations

### Model Performance Comparison
![Model Comparison](images/05_model_comparison.png)

### Feature Importance
![Feature Importance](images/08_feature_importance.png)

### Residual Analysis
![Residuals](images/06_residual_analysis.png)

## 🔍 Top Predictive Features

1. **Bedrooms** (0.213) - Number of bedrooms
2. **SquareFeet** (0.185) - Total living area
3. **Condition** (0.058) - Property condition rating
4. **DistanceToCenter** (0.058) - Miles from city center
5. **TotalRooms** (0.054) - Combined bed + bath count
6. **Neighborhood** (various) - Location significantly impacts price

## 🏗️ Feature Engineering

New features created to improve predictions:

| Feature | Description |
|---------|-------------|
| `TotalRooms` | Bedrooms + Bathrooms |
| `BathBedRatio` | Bathrooms / Bedrooms |
| `IsNewConstruction` | Built within 5 years |
| `IsRecentlyRenovated` | Renovated within 5 years |
| `HasPremiumAmenities` | Pool, fireplace, or 2+ car garage |
| `GoodSchoolDistrict` | School rating ≥ 8 |
| `LowCrimeArea` | Crime rate ≤ 10 |
| `SqFtCategory` | Size classification |

## 🛠️ Technologies Used

- **Python 3.12**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - ML algorithms, preprocessing, evaluation
- **XGBoost** - Gradient boosting
- **Matplotlib, Seaborn** - Visualizations
- **SciPy** - Statistical analysis

## 🚀 How to Use

### Run the Analysis
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
jupyter notebook house_price_regression.ipynb
```

### Load & Use the Trained Model
```python
import pickle
import pandas as pd

# Load model
with open('models/xgboost_house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature columns
with open('models/feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Prepare your property data as DataFrame with same columns
# property_data = pd.DataFrame([{...}])
# predicted_price = model.predict(property_data[feature_cols])[0]
```

## 💼 Business Applications

1. **Price Estimation** - Instant property valuations for listings
2. **Investment Analysis** - Identify undervalued properties
3. **Renovation ROI** - Quantify value added by improvements
4. **Market Comparables** - Data-driven comp analysis
5. **Portfolio Valuation** - Bulk property assessments

## 📌 Model Insights

### What Increases Property Value:
- ✅ More bedrooms and bathrooms
- ✅ Larger square footage
- ✅ Better property condition
- ✅ Proximity to city center
- ✅ Good school districts
- ✅ Premium amenities (pool, garage)

### What Decreases Property Value:
- ❌ Higher crime rates
- ❌ Poor property condition
- ❌ Industrial neighborhoods
- ❌ Older construction without renovation

## 🔗 Related Projects

- **[Project 1: EDA](../01-exploratory-data-analysis/)** - Customer churn analysis
- **[Project 2: Classification](../02-classification-ml/)** - Churn prediction model

## 👤 Author

**Alexy Louis**  
Data Analyst

📧 alexy.louis.scholar@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/alexy-louis-19a5a9262/)  
💻 [GitHub](https://github.com/Smooth-Cactus0)

---

*This project is part of my Data Analysis Portfolio. See the [main repository](../) for more projects.*
