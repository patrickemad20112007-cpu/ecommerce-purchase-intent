# 🛒 E-Commerce Real-Time Purchase Intent & Conversion Engine

> **Predicting e-commerce visitor purchase intent in real-time to automate dynamic marketing interventions (exit-intent coupons vs live chat) while preventing margin cannibalization on organic buyers.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Repository-purple.svg)](https://archive.ics.uci.edu/dataset/468)

---

## 🚀 Live Demo & Web App

- **Live Web App:** [Try the Live Streamlit App on Streamlit Cloud](https://share.streamlit.io/) *(Replace with deployed URL)*
- **Interactive Capabilities:**
  1. **Real-Time Session Simulator:** Test presets (Hot Buyer, Hesitant Abandoner, Window Shopper) or input custom session browsing patterns.
  2. **Automated Action Prescription:** Instant classification into Margin Protection (0% discount), Dynamic Exit-Intent Voucher (15%), or Passive Exploration.
  3. **Campaign ROI Calculator (Stretch Goal):** Model monthly sessions, average order value, margin, and discount costs to forecast net monthly incremental profit.

---

## 📊 Benchmark Results (Untouched 20% Test Set)

Models evaluated on **2,466 untouched held-out sessions** with 5-Fold Stratified Cross-Validation on the training split:

| Model | ROC-AUC | PR-AUC (Avg Prec) | Buyer Recall | Accuracy | Business Feasibility & Utility |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Dummy Majority)** | 0.5000 | 0.1547 | 0.00% | 84.53% | **Zero Value:** Predicts no buyers, captures zero revenue |
| **Logistic Regression (Balanced)** | 0.9058 | 0.6680 | 76.08% | 85.44% | Linear baseline; misses complex page interaction ratios |
| **Random Forest (Balanced)** | 0.9300 | 0.7361 | 82.31% | 87.35% | Strong non-linear ensemble; slightly slower inference |
| **Tuned LightGBM (Champion)** | **0.9329** | **0.7391** | **84.03%** | **85.93%** | **Optimal:** Highest PR-AUC, catches 321 of 382 buyers with <5ms latency |

> **Key Evaluation Note:** Because only 15.5% of visitors convert, standard **Accuracy is deceptive**. A dummy model predicting "no purchase" scores 84.5% accuracy but catches zero buyers. **PR-AUC and Recall** are the primary commercial drivers.

---

## 💡 The Business Decision Policy

Rather than treating every visitor the same or offering blanket discounts, our trained engine categorizes active sessions into three actionable business tiers:

```
                            [ Live Browsing Session ]
                                        │
                           (Tuned LightGBM Inference)
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   Intent ≥ 70%                 35% ≤ Intent < 70%              Intent < 35%
 [ High-Intent Buyer ]        [ Hesitant Abandoner ]        [ Casual Explorer ]
           │                            │                            │
   ✅ PROTECT MARGIN            🎁 AUTOMATE CONVERSION         ⚪ PASSIVE BROWSING
 • Do NOT offer discounts     • Trigger 15% exit popup       • Show popular catalog
 • Avoid cannibalization      • Proactive Live Chat assist   • Capture newsletter
```

---

## 🔍 Key Data & Engineering Insights

1. **The PageValue Filter:** Visitors viewing pages with `PageValues > 0` are **6.5x more likely to purchase**. Non-buyers have a median `PageValue` of `0.0`.
2. **Behavioral Feature Engineering:**
   - `Product_Page_Ratio = ProductRelated / Total_Pages`: Captures product exploration density.
   - `PageValue_Bounce_Ratio = PageValues / (BounceRates + 1e-4)`: Penalizes high-value sessions showing sudden bounce friction.
3. **Leakage-Free Guarantee:** Stratified 80/20 train/test split executed **prior** to any data transformation, scaling, or target encoding.

---

## 🛠️ Project Structure

```text
ecommerce-purchase-intent/
├── data/
│   └── online_shoppers_intention.csv    # UCI dataset (12,330 sessions)
├── models/
│   └── best_pipeline.joblib             # Serialized Scikit-Learn + LightGBM pipeline
├── notebooks/
│   └── 01_purchase_intent_analysis.ipynb# Clean, 8-step end-to-end Jupyter Notebook
├── reports/
│   ├── figures/                         # High-res charts (Confusion matrix, ROC, PR, EDA)
│   ├── metrics.json                     # Verified benchmark scores & confusion matrix
│   ├── PRESENTATION.md                  # 12-slide executive presentation outline
│   └── LINKEDIN_AND_CV.md               # Ready-to-use CV bullet and LinkedIn post
├── train_pipeline.py                    # Training, cross-validation & tuning script
├── generate_notebook.py                 # Notebook generator script
├── app.py                               # Production Streamlit Web Application
├── requirements.txt                     # Project dependencies
└── README.md                            # Executive project documentation
```

---

## ⚡ How to Run Locally

### 1. Clone the repository & enter directory:
```bash
git clone https://github.com/your-username/ecommerce-purchase-intent.git
cd ecommerce-purchase-intent
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the model pipeline:
```bash
python train_pipeline.py
```

### 4. Launch the Streamlit Web Application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📦 Dataset Reference

- **Source:** [UCI Machine Learning Repository: Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468)
- **Citation:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications.

---

## 👤 Author

Developed for **NextGen Institute Machine Learning Program**  
*Feel free to connect on LinkedIn or open an issue for questions or enhancements!*
