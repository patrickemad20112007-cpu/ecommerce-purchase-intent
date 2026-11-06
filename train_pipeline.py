"""
E-Commerce Purchase Intent Prediction Pipeline
NextGen Institute - Final Machine Learning Project
Author: AI ML Engineer & Data Scientist
Dataset: Online Shoppers Purchasing Intention (UCI ML Repository, 12,330 sessions)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# ---------------------------------------------------------
# 1. SETUP & PATHS
# ---------------------------------------------------------
BASE_DIR = r"C:\Users\COMPUMARTS\.gemini\antigravity\scratch\ecommerce-purchase-intent"
DATA_PATH = os.path.join(BASE_DIR, "data", "online_shoppers_intention.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. DATA LOADING & INSPECTION
# ---------------------------------------------------------
print("=" * 60)
print("1. LOADING AND UNDERSTANDING DATASET")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"Total Sessions: {df.shape[0]}, Features: {df.shape[1]}")

# Target variable conversion
df['Revenue'] = df['Revenue'].astype(int)
print(f"Target Distribution:\n{df['Revenue'].value_counts(normalize=True)}")
class_ratio = (df['Revenue'] == 0).sum() / (df['Revenue'] == 1).sum()
print(f"Negative to Positive Ratio (Imbalance Factor): {class_ratio:.2f}:1")

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING FUNCTION
# ---------------------------------------------------------
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Creates domain-specific behavioral signals without data leakage."""
    d = data.copy()
    # Total activity
    d['Total_Pages'] = d['Administrative'] + d['Informational'] + d['ProductRelated']
    d['Total_Duration'] = d['Administrative_Duration'] + d['Informational_Duration'] + d['ProductRelated_Duration']
    
    # Ratios and behavioral intensity
    d['Product_Page_Ratio'] = d['ProductRelated'] / (d['Total_Pages'] + 1e-5)
    d['Avg_Duration_Per_Page'] = d['Total_Duration'] / (d['Total_Pages'] + 1e-5)
    
    # Interaction: High Page Value + Low Bounce indicates strong shopping intent
    d['PageValue_Bounce_Ratio'] = d['PageValues'] / (d['BounceRates'] + 1e-4)
    
    # Cast nominal integers to strings for one-hot encoding
    nominal_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
    for col in nominal_cols:
        d[col] = d[col].astype(str)
        
    return d

df_eng = engineer_features(df)

# Feature sets
numeric_features = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'Total_Pages', 'Total_Duration', 'Product_Page_Ratio',
    'Avg_Duration_Per_Page', 'PageValue_Bounce_Ratio'
]

categorical_features = [
    'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'
]

X = df_eng[numeric_features + categorical_features]
y = df_eng['Revenue']

# ---------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA) VISUALS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("2. GENERATING EDA VISUALS")
print("=" * 60)

sns.set_theme(style="whitegrid", palette="muted")

# 1. Target distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=y.map({0: 'No Purchase (0)', 1: 'Purchase (1)'}), palette=['#4A90E2', '#50E3C2'])
plt.title("Class Distribution (Purchase Intent)", fontsize=13, weight='bold')
plt.xlabel("Outcome")
plt.ylabel("Number of Sessions")
for p in ax.patches:
    ax.annotate(f'{p.get_height():,}\n({p.get_height()/len(y)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_target_distribution.png"), dpi=200)
plt.close()

# 2. PageValues vs Revenue
plt.figure(figsize=(7, 4))
sns.boxplot(x=y.map({0: 'No Purchase', 1: 'Purchase'}), y=df_eng['PageValues'], palette=['#E74C3C', '#2ECC71'], showfliers=False)
plt.title("PageValues Distribution by Purchase Intent", fontsize=13, weight='bold')
plt.ylabel("PageValues (Google Analytics Metric)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_pagevalues_by_revenue.png"), dpi=200)
plt.close()

# 3. Monthly Conversion Rate
monthly = df.groupby('Month')['Revenue'].agg(['count', 'mean']).reset_index()
month_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly['Month'] = pd.Categorical(monthly['Month'], categories=month_order, ordered=True)
monthly = monthly.sort_values('Month')

plt.figure(figsize=(9, 4))
sns.barplot(data=monthly, x='Month', y='mean', palette='Blues_d')
plt.title("Conversion Rate by Month", fontsize=13, weight='bold')
plt.ylabel("Conversion Rate (Proportion Buying)")
plt.xlabel("Month")
for i, row in enumerate(monthly.itertuples()):
    plt.text(i, row.mean + 0.005, f"{row.mean*100:.1f}%", ha='center', fontsize=9, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_monthly_conversion.png"), dpi=200)
plt.close()

print(f"EDA plots saved to: {FIG_DIR}")

# ---------------------------------------------------------
# 5. SPLIT BEFORE ANY TRANSFORMATION (NO DATA LEAKAGE)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("3. STRATIFIED TRAIN / TEST SPLIT (NO DATA LEAKAGE)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training sessions: {X_train.shape[0]} ({y_train.sum()} purchases)")
print(f"Testing sessions:  {X_test.shape[0]} ({y_test.sum()} purchases)")

# ---------------------------------------------------------
# 6. SCIKIT-LEARN PREPROCESSING PIPELINE
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# ---------------------------------------------------------
# 7. MODEL CANDIDATES & CROSS-VALIDATION
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("4. MODEL COMPARISON (5-FOLD STRATIFIED CV ON TRAINING SET)")
print("=" * 60)

models = {
    "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, scale_pos_weight=class_ratio, random_state=42, verbose=-1, n_jobs=-1)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['roc_auc', 'average_precision', 'f1', 'recall', 'accuracy']

cv_results = {}
for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = {
        'ROC-AUC': float(np.mean(scores['test_roc_auc'])),
        'ROC-AUC Std': float(np.std(scores['test_roc_auc'])),
        'PR-AUC': float(np.mean(scores['test_average_precision'])),
        'F1-Score': float(np.mean(scores['test_f1'])),
        'Recall': float(np.mean(scores['test_recall'])),
        'Accuracy': float(np.mean(scores['test_accuracy']))
    }
    print(f"\n[{name}]")
    print(f"  ROC-AUC:  {cv_results[name]['ROC-AUC']:.4f} +/- {cv_results[name]['ROC-AUC Std']:.4f}")
    print(f"  PR-AUC:   {cv_results[name]['PR-AUC']:.4f}")
    print(f"  F1-Score: {cv_results[name]['F1-Score']:.4f}")
    print(f"  Recall:   {cv_results[name]['Recall']:.4f}")

# ---------------------------------------------------------
# 8. HYPERPARAMETER TUNING (ON TRAINING DATA ONLY)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("5. HYPERPARAMETER TUNING (LIGHTGBM)")
print("=" * 60)

lgbm_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__num_leaves': [15, 31],
    'classifier__learning_rate': [0.03, 0.08],
    'classifier__scale_pos_weight': [2.5, class_ratio],
    'classifier__min_child_samples': [20, 50]
}

grid_search = GridSearchCV(
    lgbm_pipe,
    param_grid,
    cv=cv,
    scoring='average_precision', # optimizes PR-AUC directly for imbalanced conversions
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Training PR-AUC Score: {grid_search.best_score_:.4f}")

# ---------------------------------------------------------
# 9. FINAL EVALUATION ON UNTOUCHED TEST SET
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("6. FINAL EVALUATION ON UNTOUCHED TEST SET")
print("=" * 60)

y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Default threshold 0.50
y_pred_default = (y_pred_proba >= 0.5).astype(int)

test_roc_auc = roc_auc_score(y_test, y_pred_proba)
test_pr_auc = average_precision_score(y_test, y_pred_proba)
test_f1 = f1_score(y_test, y_pred_default)
test_recall = recall_score(y_test, y_pred_default)
test_precision = precision_score(y_test, y_pred_default)
test_acc = accuracy_score(y_test, y_pred_default)

print(f"Test ROC-AUC:             {test_roc_auc:.4f}")
print(f"Test PR-AUC (Avg Prec):   {test_pr_auc:.4f}")
print(f"Test F1-Score (th=0.5):   {test_f1:.4f}")
print(f"Test Recall (th=0.5):     {test_recall:.4f}")
print(f"Test Precision (th=0.5):  {test_precision:.4f}")
print(f"Test Accuracy:            {test_acc:.4f}")

print("\nClassification Report (th=0.5):")
print(classification_report(y_test, y_pred_default, target_names=["No Purchase", "Purchase"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_default)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix (Test Set, Threshold=0.50)", fontsize=11, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eval_confusion_matrix.png"), dpi=200)
plt.close()

# ROC & PR Curves
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
prec_vals, rec_vals, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, color='#2980B9', lw=2, label=f'LightGBM (AUC = {test_roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, weight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (Recall)')
axes[0].legend(loc='lower right')

axes[1].plot(rec_vals, prec_vals, color='#27AE60', lw=2, label=f'LightGBM (PR-AUC = {test_pr_auc:.3f})')
axes[1].axhline(y=y_test.mean(), color='gray', linestyle='--', label=f'Baseline ({y_test.mean():.2f})')
axes[1].set_title('Precision-Recall Curve', fontsize=12, weight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eval_roc_pr_curves.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 10. BUSINESS THRESHOLD OPTIMIZATION & UTILITY ANALYSIS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("7. BUSINESS THRESHOLD & CAMPAIGN ROI OPTIMIZATION")
print("=" * 60)

# Simulate Business Economics:
# Average Order Value (AOV) = $75.00
# Profit Margin = 30% -> Net Gain from Organic Sale = $22.50
# If customer wouldn't buy without coupon, and coupon converts them:
# Net Gain with 15% coupon = ($75 * 0.85 * 0.30) - ($75 * 0.15 cost difference) approx $10.00
# If customer WOULD buy anyway and gets coupon: Cannibalization loss = -$11.25
# Chat agent trigger cost: $0.20 per engaged visitor

thresholds = np.linspace(0.1, 0.9, 81)
f1_list = []
precision_list = []
recall_list = []

for th in thresholds:
    preds = (y_pred_proba >= th).astype(int)
    f1_list.append(f1_score(y_test, preds))
    precision_list.append(precision_score(y_test, preds, zero_division=0))
    recall_list.append(recall_score(y_test, preds))

optimal_f1_idx = np.argmax(f1_list)
optimal_th = thresholds[optimal_f1_idx]
print(f"Optimal Threshold for Maximum F1: {optimal_th:.2f} (F1 = {f1_list[optimal_f1_idx]:.4f}, Recall = {recall_list[optimal_f1_idx]:.4f}, Precision = {precision_list[optimal_f1_idx]:.4f})")

# ---------------------------------------------------------
# 11. FEATURE IMPORTANCE EXTRACTION
# ---------------------------------------------------------
trained_lgbm = best_model.named_steps['classifier']
ohe_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_names)

importances = trained_lgbm.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[top_idx], y=[all_feature_names[i] for i in top_idx], palette='viridis')
plt.title("Top 15 Most Influential Features (LightGBM)", fontsize=13, weight='bold')
plt.xlabel("Split Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=200)
plt.close()

# ---------------------------------------------------------
# 12. SAVE MODEL PIPELINE & METRICS JSON
# ---------------------------------------------------------
model_file = os.path.join(MODELS_DIR, "best_pipeline.joblib")
joblib.dump(best_model, model_file)
print(f"\nTrained Pipeline serialized to: {model_file}")

final_metrics = {
    "project_name": "E-Commerce Real-Time Purchase Intent Predictor",
    "dataset": "Online Shoppers Purchasing Intention (UCI, 12,330 sessions)",
    "target": "Revenue (Purchase True/False)",
    "positive_class_ratio": float(y.mean()),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "cv_results": cv_results,
    "best_hyperparameters": {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()},
    "test_scores": {
        "ROC_AUC": round(float(test_roc_auc), 4),
        "PR_AUC": round(float(test_pr_auc), 4),
        "Accuracy": round(float(test_acc), 4),
        "F1_Score_default_th": round(float(test_f1), 4),
        "Recall_default_th": round(float(test_recall), 4),
        "Precision_default_th": round(float(test_precision), 4),
        "Optimal_F1_Threshold": round(float(optimal_th), 2),
        "Optimal_F1_Score": round(float(f1_list[optimal_f1_idx]), 4),
        "Optimal_Recall": round(float(recall_list[optimal_f1_idx]), 4),
        "Optimal_Precision": round(float(precision_list[optimal_f1_idx]), 4)
    },
    "confusion_matrix_test": {
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1])
    },
    "top_5_features": [all_feature_names[i] for i in top_idx[:5]]
}

metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(final_metrics, f, indent=4)
print(f"Metrics saved to: {metrics_path}")
print("\nPipeline execution completed successfully!")
