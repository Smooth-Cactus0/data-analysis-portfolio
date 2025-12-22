"""
Classical ML Training Script - Trains and evaluates all baseline models.
Author: Alexy Louis
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from scripts.synthetic_data import create_imdb_like_dataset
from src.preprocessing import preprocess_texts

# Paths
MODELS_DIR = Path('models')
IMAGES_DIR = Path('images')
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

print("="*60)
print("CLASSICAL ML TRAINING PIPELINE")
print("="*60)

# 1. Generate data
print("\n[1/5] Generating synthetic dataset...")
dataset = create_imdb_like_dataset(n_train=25000, n_test=25000, seed=42)
train_texts = preprocess_texts(dataset['train']['text'])
test_texts = preprocess_texts(dataset['test']['text'])
y_train = np.array(dataset['train']['label'])
y_test = np.array(dataset['test']['label'])
print(f"  Train: {len(train_texts):,} | Test: {len(test_texts):,}")

# 2. Vectorize
print("\n[2/5] TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=3, max_df=0.9)
X_train = tfidf.fit_transform(train_texts)
X_test = tfidf.transform(test_texts)
joblib.dump(tfidf, MODELS_DIR / 'tfidf_vectorizer.joblib')
print(f"  Features: {X_train.shape[1]:,}")

# 3. Train models
print("\n[3/5] Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Multinomial NB': MultinomialNB(alpha=0.1),
    'Complement NB': ComplementNB(alpha=0.1),
    'Linear SVM': LinearSVC(C=0.5, max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results = []
for name, model in models.items():
    print(f"  Training {name}...", end=' ')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc, 'F1': f1})
    joblib.dump(model, MODELS_DIR / f'{name.lower().replace(" ", "_")}.joblib')
    print(f"Acc: {acc:.4f}")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print("\n" + results_df.to_string(index=False))

# 4. Visualizations
print("\n[4/5] Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('viridis', len(results_df))
bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors)
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_title('Classical ML Model Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0.8, 0.9)
for bar, acc in zip(bars, results_df['Accuracy']):
    ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2, f'{acc:.2%}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '01_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Best model confusion matrix
best_model_name = results_df.iloc[0]['Model']
best_model = joblib.load(MODELS_DIR / f'{best_model_name.lower().replace(" ", "_")}.joblib')
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(IMAGES_DIR / '02_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
for name in ['Logistic Regression', 'Complement NB', 'Linear SVM']:
    model = joblib.load(MODELS_DIR / f'{name.lower().replace(" ", "_")}.joblib')
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Classical ML Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(IMAGES_DIR / '03_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Save sample data
print("\n[5/5] Saving sample data...")
sample_df = pd.DataFrame({
    'text': dataset['train']['text'][:500],
    'label': dataset['train']['label'][:500]
})
sample_df.to_csv('data/sample/reviews_sample.csv', index=False)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.2%}")
print(f"\nFiles saved:")
print(f"  - models/*.joblib (6 models + vectorizer)")
print(f"  - images/*.png (3 visualizations)")
print(f"  - data/sample/reviews_sample.csv")
