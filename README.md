# 📊 Data Analysis Portfolio

**Comprehensive collection of data analysis, machine learning, and visualization projects**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 👤 Author

**Alexy Louis**
- 📧 Email: alexy.louis.scholar@gmail.com
- 💼 LinkedIn: [linkedin.com/in/alexy-louis-19a5a9262](https://www.linkedin.com/in/alexy-louis-19a5a9262/)
- 🐙 GitHub: [github.com/Smooth-Cactus0](https://github.com/Smooth-Cactus0)

---

## 🗂️ Projects Overview

| # | Project | Type | Key Techniques | Highlights |
|---|---------|------|----------------|------------|
| 1 | [Customer Churn EDA](./01-exploratory-data-analysis/) | Exploratory Analysis | Statistical Analysis, Visualization | 5,000 records, 14 visualizations |
| 2 | [Churn Prediction (Classification)](./02-classification-ml/) | Machine Learning | Logistic Regression, Random Forest, XGBoost | ROC-AUC 0.79, 4 models |
| 3 | [House Price Prediction (Regression)](./03-regression-ml/) | Machine Learning | Ridge, Lasso, XGBoost, Stacking | R² 0.937, 7 models |
| 4 | [ETL Data Pipeline](./04-data-processing-apis/) | Data Engineering | ETL, Validation, Transformation | 6 sources, 15+ validations |
| 5 | [Energy Consumption Forecasting](./05-time-series-forecasting/) | Time Series | Prophet, LSTM, LightGBM, Ensemble | **MAPE 2.18%**, 17K records |
| 6 | [NLP Sentiment Analysis](./06-nlp-sentiment-analysis/) | NLP / Deep Learning | BiLSTM, CNN, DistilBERT | **87.7% accuracy**, 3 approaches |

---

## 📁 Project Details

### 1️⃣ Customer Churn Exploratory Data Analysis
**[01-customer-churn-eda/](./01-customer-churn-eda/)**

Deep dive into telecom customer churn patterns with comprehensive statistical analysis.

- **Dataset**: 5,000 customers, 20+ features
- **Techniques**: Distribution analysis, correlation, segmentation
- **Outputs**: 14 publication-ready visualizations
- **Key Finding**: Month-to-month contracts have 3x higher churn rate

### 2️⃣ Customer Churn Prediction (Classification)
**[02-churn-prediction-ml/](./02-churn-prediction-ml/)**

Production-ready ML pipeline for predicting customer churn.

- **Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Best Performance**: XGBoost with ROC-AUC 0.79
- **Features**: Feature importance analysis, threshold optimization
- **Business Impact**: Identify 79% of churners before they leave

### 3️⃣ House Price Prediction (Regression)
**[03-house-price-regression/](./03-house-price-regression/)**

Advanced regression modeling for real estate price prediction.

- **Dataset**: 1,460 houses, 80 features
- **Models**: Ridge, Lasso, ElasticNet, Random Forest, XGBoost, Stacking
- **Best Performance**: Stacking Ensemble with R² = 0.937
- **Key Insight**: Top predictors are OverallQual, GrLivArea, GarageCars

### 4️⃣ Multi-Source ETL Data Pipeline
**[04-data-processing-apis/](./04-data-processing-apis/)**

Production-grade data engineering framework demonstrating ETL best practices.

- **Components**: DataLoader, DataValidator, DataTransformer, PipelineOrchestrator
- **Data Sources**: CSV, JSON, API integration
- **Validations**: 15+ rules including referential integrity
- **Features**: Chainable transformations, audit trails, comprehensive logging

### 5️⃣ Energy Consumption Forecasting
**[05-time-series-forecasting/](./05-time-series-forecasting/)**

Comprehensive time series analysis with multiple forecasting approaches.

- **Dataset**: 17,497 hourly records (2 years), 138 engineered features
- **Models**: ARIMA, SARIMA, Prophet, LightGBM, XGBoost, LSTM
- **Best Performance**: Ensemble (LightGBM + XGBoost) with **MAPE 2.18%**
- **Features**: Multiple seasonalities, weather integration, anomaly detection
- **Key Insight**: Tree-based ensembles outperform deep learning for this domain

### 6️⃣ NLP Sentiment Analysis ⭐ NEW
**[06-nlp-sentiment-analysis/](./06-nlp-sentiment-analysis/)**

Progressive NLP pipeline from classical ML to transformers for sentiment classification.

- **Dataset**: 50,000 synthetic movie reviews (balanced positive/negative)
- **Approach 1 - Classical ML**: Logistic Regression, SVM, Naive Bayes → 87% accuracy
- **Approach 2 - Deep Learning**: BiLSTM, CNN → **87.7% accuracy**
- **Approach 3 - Transformers**: DistilBERT fine-tuning → 87.6% accuracy
- **Features**: GPU-accelerated training, comprehensive visualizations
- **Key Insight**: BiLSTM and CNN match transformer performance on this task

---

## 🛠️ Tech Stack

### Languages & Frameworks
- **Python 3.12** - Primary language
- **Jupyter Notebooks** - Interactive analysis
- **SQL** - Data querying

### Data Processing
- pandas, NumPy, scipy
- statsmodels (time series)

### Machine Learning
- scikit-learn
- XGBoost, LightGBM
- TensorFlow/Keras (LSTM)
- Prophet (forecasting)

### NLP & Deep Learning
- PyTorch
- Hugging Face Transformers
- NLTK

### Visualization
- Matplotlib, Seaborn
- Plotly (interactive)

### Data Engineering
- Custom ETL framework
- python-docx, openpyxl

---

## 📈 Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Data Analysis** | EDA, Statistical Testing, Feature Engineering |
| **Machine Learning** | Classification, Regression, Time Series, Ensemble Methods |
| **Deep Learning** | LSTM, BiLSTM, CNN, Sequence Modeling |
| **NLP** | Text Classification, Sentiment Analysis, Transformer Fine-tuning |
| **Data Engineering** | ETL Pipelines, Data Validation, Transformation |
| **Visualization** | Matplotlib, Seaborn, Publication-Ready Charts |
| **Software Engineering** | OOP, Modular Design, Documentation |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Smooth-Cactus0/data-analysis-portfolio.git
cd data-analysis-portfolio

# Install dependencies
pip install -r requirements.txt

# Navigate to any project
cd 05-time-series-forecasting
jupyter notebook
```

---

## 📊 Portfolio Statistics

| Metric | Value |
|--------|-------|
| **Total Projects** | 6 |
| **Total Records Analyzed** | 75,000+ |
| **Models Trained** | 30+ |
| **Visualizations Created** | 60+ |
| **Lines of Code** | 7,000+ |

---

## 📄 License

This portfolio is available for learning and reference purposes. Please cite if used.

---

*Last Updated: December 2025*
