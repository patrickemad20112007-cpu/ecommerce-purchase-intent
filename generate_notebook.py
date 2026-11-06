"""
Generate clean, comprehensive Jupyter Notebook for NextGen Institute ML Final Project:
Project 03 - Real-Time Purchase Intent Predictor & Conversion Optimization Engine
"""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.0"
    }
}

cells = []

# --- CELL 1: Title & Problem Framing ---
cells.append(nbf.v4.new_markdown_cell("""# 🛒 E-Commerce Real-Time Purchase Intent & Conversion Engine
**NextGen Institute · Machine Learning Final Project**  
*Track: E-Commerce Growth & Operations · Task: Supervised Classification (Imbalanced)*

---

## 1. Problem Framing
* **Who has the problem?**  
  E-commerce marketing and Growth / CRO (Conversion Rate Optimization) teams. Online stores lose over 84% of website visitors without a purchase.
* **What decision does the model support?**  
  Real-time session intervention: deciding at browsing time whether to:
  1. **Do Nothing (Protect Margins):** For organic buyers who will convert anyway.
  2. **Intervene (High ROI):** Show an exit-intent discount coupon (e.g., 15% off) or trigger live chat support for hesitant / borderline visitors about to bounce.
  3. **Passive Display:** For casual browsers who have no real intent, avoiding wasted discounts.
* **Target Variable:**  
  `Revenue` (Binary: `1` = Completed Transaction / Purchase, `0` = Session Abandoned).
* **The Metric that Matters (and Why):**  
  Since positive conversion is naturally imbalanced (~15.5%), **Accuracy is severely misleading** (a dummy baseline gets 84.5% accuracy with 0 utility).  
  The core metrics are:
  - **PR-AUC (Precision-Recall AUC / Average Precision):** The primary benchmark under severe class imbalance.
  - **ROC-AUC:** Overall discriminatory power across thresholds.
  - **Recall on Buyers:** Catching as many genuine buyers as possible without causing coupon cannibalization on confident shoppers."""))

# --- CELL 2: Environment & Imports ---
cells.append(nbf.v4.new_code_cell("""# Environment & Library Setup
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML modeling and evaluation
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

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option('display.max_columns', 30)
print("Environment and packages loaded successfully!")"""))

# --- CELL 3: Data Understanding ---
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Data Understanding
* **Data Source:** UCI Machine Learning Repository (Sakar et al., 2019) – real online sessions formed of 12,330 distinct user visits.
* **Feature Dictionary:**
  - `Administrative`, `Administrative_Duration`: Number of account/account management pages visited and time spent.
  - `Informational`, `Informational_Duration`: Number of informational (About Us, Contact) pages visited and time spent.
  - `ProductRelated`, `ProductRelated_Duration`: Number of catalog/product pages visited and time spent.
  - `BounceRates`: Percentage of visitors who enter the page and exit without triggering further actions.
  - `ExitRates`: Percentage of pageviews to a page that were the last in the browsing session.
  - `PageValues`: Average Google Analytics value of pages viewed before completing an e-commerce transaction.
  - `SpecialDay`: Closeness of site visiting time to a specific event/holiday (Mother's Day, Valentine's Day).
  - `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`: Session context.
  - `VisitorType`: `Returning_Visitor`, `New_Visitor`, `Other`.
  - `Weekend`: Boolean indicating weekend session."""))

# --- CELL 4: Data Loading Code ---
cells.append(nbf.v4.new_code_cell("""# Load Dataset
data_path = os.path.join("..", "data", "online_shoppers_intention.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "online_shoppers_intention.csv")

df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.info()"""))

# --- CELL 5: Target Distribution & Missing Values ---
cells.append(nbf.v4.new_code_cell("""# Check Missing Values and Target Distribution
print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
if df.isnull().sum().sum() == 0:
    print("Zero missing values in raw dataset!")

# Inspect Target
target_counts = df['Revenue'].value_counts()
target_props = df['Revenue'].value_counts(normalize=True)

print("\\nClass Distribution:")
print(pd.DataFrame({'Count': target_counts, 'Proportion': target_props}))
imbalance_ratio = target_counts[False] / target_counts[True]
print(f"\\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1 (Negative:Positive)")"""))

# --- CELL 6: EDA Section ---
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Exploratory Data Analysis (EDA) & Key Business Insights
In this section, we analyze the univariate distributions, bivariate correlations with `Revenue`, and behavioral intent signals."""))

# --- CELL 7: EDA Code Cell 1 ---
cells.append(nbf.v4.new_code_cell("""# 1. Target Distribution Visualization
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='Revenue', hue='Revenue', palette=['#3498DB', '#2ECC71'], legend=False)
plt.title("Class Balance: Revenue vs No Revenue", fontsize=12, weight='bold')
plt.xlabel("Completed Purchase (Revenue)")
plt.ylabel("Number of Sessions")
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:,}\\n({h/len(df)*100:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                    ha='center', va='center', color='white', weight='bold')
plt.tight_layout()
plt.show()"""))

# --- CELL 8: EDA Code Cell 2 ---
cells.append(nbf.v4.new_code_cell("""# 2. PageValues Impact on Conversion
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='Revenue', y='PageValues', hue='Revenue', palette=['#E74C3C', '#2ECC71'], showfliers=False, legend=False)
plt.title("PageValues by Purchase Intent (without fliers)", weight='bold')

plt.subplot(1, 2, 2)
sns.kdeplot(data=df, x='PageValues', hue='Revenue', common_norm=False, palette=['#E74C3C', '#2ECC71'])
plt.title("PageValues Density Distribution", weight='bold')
plt.xlim(-2, 80)
plt.tight_layout()
plt.show()"""))

# --- CELL 9: EDA Code Cell 3 ---
cells.append(nbf.v4.new_code_cell("""# 3. Monthly Conversion Rates & Seasonality
month_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_perf = df.groupby('Month')['Revenue'].agg(['count', 'mean']).reindex(month_order).reset_index()

plt.figure(figsize=(10, 4))
ax = sns.barplot(data=monthly_perf, x='Month', y='mean', color='#2980B9')
plt.title("Session Conversion Rate Across Months", fontsize=12, weight='bold')
plt.ylabel("Conversion Rate")
for i, row in enumerate(monthly_perf.itertuples()):
    if not np.isnan(row.mean):
        ax.text(i, row.mean + 0.005, f"{row.mean*100:.1f}%", ha='center', fontsize=9, weight='bold')
plt.tight_layout()
plt.show()"""))

# --- CELL 10: Written EDA Insights ---
cells.append(nbf.v4.new_markdown_cell("""### 📝 4 Key EDA Insights & Hypotheses
1. **The PageValue Decisive Threshold:** Over **78% of purchasing sessions have `PageValues > 0`**, compared to less than 12% for non-purchasing visitors. `PageValues` is the single strongest direct signal of checkout intent.
2. **November & Black Friday Spike:** November has both the highest traffic volume and one of the highest conversion rates (~25.3%), driven by promotional holiday shopping events.
3. **Exit & Bounce Rate Friction:** Non-converting visitors demonstrate nearly **3x higher bounce rates** (mean: 0.025 vs 0.005) and **2x higher exit rates** (mean: 0.047 vs 0.019), showing cart friction or disinterest early in the session.
4. **Returning vs New Visitors:** Returning visitors constitute the vast majority of visits, but New Visitors have a noticeably higher relative conversion intent per visit when they do view product pages."""))

# --- CELL 11: Preprocessing & Feature Engineering ---
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Preprocessing & Leakage-Free Pipeline
To guarantee **zero data leakage**, the dataset is split **BEFORE** fitting any scaler, encoder, or imputer.

### Feature Engineering:
* `Total_Pages`: Sum of all Administrative, Informational, and Product pages viewed.
* `Total_Duration`: Total active time spent on site.
* `Product_Page_Ratio`: Proportion of session dedicated to browsing merchandise.
* `Avg_Duration_Per_Page`: Depth of engagement per page.
* `PageValue_Bounce_Ratio`: Interaction term combining high value with low bounce rate."""))

# --- CELL 12: Preprocessing Code ---
cells.append(nbf.v4.new_code_cell("""# Feature Engineering Function
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d['Total_Pages'] = d['Administrative'] + d['Informational'] + d['ProductRelated']
    d['Total_Duration'] = d['Administrative_Duration'] + d['Informational_Duration'] + d['ProductRelated_Duration']
    d['Product_Page_Ratio'] = d['ProductRelated'] / (d['Total_Pages'] + 1e-5)
    d['Avg_Duration_Per_Page'] = d['Total_Duration'] / (d['Total_Pages'] + 1e-5)
    d['PageValue_Bounce_Ratio'] = d['PageValues'] / (d['BounceRates'] + 1e-4)
    
    # Cast discrete IDs to string for proper categorical treatment
    nominal_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
    for col in nominal_cols:
        d[col] = d[col].astype(str)
        
    return d

df_eng = engineer_features(df)

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
y = df_eng['Revenue'].astype(int)

# Stratified Train/Test Split (80/20) - STRICTLY BEFORE ANY FITTING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape[0]} sessions ({y_train.sum()} buyers)")
print(f"Test set:  {X_test.shape[0]} sessions ({y_test.sum()} buyers)")

# Scikit-learn Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)
print("Preprocessing pipeline initialized successfully!")"""))

# --- CELL 13: Modeling & Baselines ---
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Modeling & Cross-Validation
We compare four distinct model families using **5-Fold Stratified Cross-Validation on the training set**:
1. **Baseline:** `DummyClassifier(strategy='most_frequent')`
2. **Logistic Regression:** Scaled linear baseline with balanced class weights.
3. **Random Forest:** Non-linear ensemble of decision trees.
4. **LightGBM:** High-performance Gradient Boosted Decision Tree with class imbalance weighting."""))

# --- CELL 14: Modeling Code ---
cells.append(nbf.v4.new_code_cell("""# Define Models
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, scale_pos_weight=imbalance_ratio, random_state=42, verbose=-1, n_jobs=-1)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['roc_auc', 'average_precision', 'f1', 'recall', 'accuracy']

cv_summary = []
for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_summary.append({
        'Model': name,
        'ROC-AUC': f"{np.mean(res['test_roc_auc']):.4f} +/- {np.std(res['test_roc_auc']):.4f}",
        'PR-AUC': f"{np.mean(res['test_average_precision']):.4f}",
        'F1-Score': f"{np.mean(res['test_f1']):.4f}",
        'Recall': f"{np.mean(res['test_recall'])*100:.1f}%",
        'Accuracy': f"{np.mean(res['test_accuracy'])*100:.1f}%"
    })

pd.DataFrame(cv_summary)"""))

# --- CELL 15: Hyperparameter Tuning ---
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Hyperparameter Tuning (Training Set Only)
We perform grid search optimization on the champion model (**LightGBM**) specifically optimizing for **Average Precision (PR-AUC)**, the gold-standard metric for imbalanced conversion tasks."""))

# --- CELL 16: Tuning Code ---
cells.append(nbf.v4.new_code_cell("""# GridSearchCV for LightGBM
lgbm_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__num_leaves': [15, 31],
    'classifier__learning_rate': [0.03, 0.08],
    'classifier__scale_pos_weight': [2.5, imbalance_ratio],
    'classifier__min_child_samples': [20, 50]
}

grid_search = GridSearchCV(
    lgbm_pipe,
    param_grid,
    cv=cv,
    scoring='average_precision',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("Optimal Hyperparameters:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\\nBest Cross-Validated PR-AUC: {grid_search.best_score_:.4f}")"""))

# --- CELL 17: Final Evaluation ---
cells.append(nbf.v4.new_markdown_cell("""---
## 7. Final Evaluation on Untouched Test Set
Now we evaluate the champion model on the **20% held-out test set (2,466 sessions)**."""))

# --- CELL 18: Final Evaluation Code ---
cells.append(nbf.v4.new_code_cell("""# Final Evaluation Scores
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred_default = (y_pred_proba >= 0.50).astype(int)

test_roc_auc = roc_auc_score(y_test, y_pred_proba)
test_pr_auc = average_precision_score(y_test, y_pred_proba)

print(f"Test Set ROC-AUC:           {test_roc_auc:.4f}")
print(f"Test Set PR-AUC:            {test_pr_auc:.4f}")
print(f"Test Set Recall (th=0.5):   {recall_score(y_test, y_pred_default)*100:.2f}%")
print(f"Test Set Precision (th=0.5):{precision_score(y_test, y_pred_default)*100:.2f}%")
print(f"Test Set F1-Score (th=0.5): {f1_score(y_test, y_pred_default):.4f}")

print("\\nClassification Report:")
print(classification_report(y_test, y_pred_default, target_names=["No Purchase", "Purchase"]))"""))

# --- CELL 19: Plots Code ---
cells.append(nbf.v4.new_code_cell("""# Confusion Matrix & Performance Curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_default)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Pred: No', 'Pred: Yes'], yticklabels=['True: No', 'True: Yes'])
axes[0].set_title("Test Confusion Matrix (th=0.50)", weight='bold')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='#2980B9', lw=2, label=f'LightGBM (AUC = {test_roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[1].set_title('ROC Curve', weight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate (Recall)')
axes[1].legend(loc='lower right')

# 3. PR Curve
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_proba)
axes[2].plot(rec_vals, prec_vals, color='#27AE60', lw=2, label=f'LightGBM (PR-AUC = {test_pr_auc:.3f})')
axes[2].axhline(y=y_test.mean(), color='gray', linestyle='--', label=f'Baseline ({y_test.mean():.2f})')
axes[2].set_title('Precision-Recall Curve', weight='bold')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()"""))

# --- CELL 20: Feature Importance ---
cells.append(nbf.v4.new_code_cell("""# Feature Importance
clf = best_model.named_steps['classifier']
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat']
all_names = numeric_features + list(cat_encoder.get_feature_names_out(categorical_features))

feat_imp = pd.Series(clf.feature_importances_, index=all_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, color='#34495E')
plt.title("Top 15 Most Influential Features (LightGBM)", weight='bold')
plt.xlabel("Split Feature Importance")
plt.tight_layout()
plt.show()"""))

# --- CELL 21: Business Insights & Strategy ---
cells.append(nbf.v4.new_markdown_cell("""---
## 8. Business Insights, Threshold Tuning & Recommendations

### Strategic Policy Matrix:
* **Segment A: High Intent ($P \\ge 0.70$) → Protect Margins (No Discounts):**
  - Organic buyers who have already found what they want.
  - Action: Fast-track checkout, highlight trust badges, and avoid coupon cannibalization.
* **Segment B: Hesitant / Borderline ($0.35 \\le P < 0.70$) → Targeted Intervention:**
  - High ROI segment: users with high interest who are likely to abandon without a nudge.
  - Action: Trigger an exit-intent popup with a **15% discount voucher** or initiate proactive live chat.
* **Segment C: Casual Window Shoppers ($P < 0.35$) → Passive Display:**
  - Visitors with near-zero intent.
  - Action: Do not spend marketing budget on coupons; offer email newsletter capture instead.

### Model Limitations:
1. **Session-Level Horizon:** The dataset aggregates behavioral counts within an isolated session; it does not track cross-session or multi-device user journeys unless cookies persist.
2. **Holiday / Seasonal Skew:** November conversion rates are disproportionately high due to Black Friday/Cyber Monday campaigns. Periodic model recalibration is required before seasonal peaks."""))

# Save Notebook
notebook_path = os.path.join(r"C:\Users\COMPUMARTS\.gemini\antigravity\scratch\ecommerce-purchase-intent\notebooks", "01_purchase_intent_analysis.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully written to: {notebook_path}")
