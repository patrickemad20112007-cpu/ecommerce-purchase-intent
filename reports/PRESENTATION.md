# Executive Presentation: Real-Time E-Commerce Purchase Intent & Conversion Engine
**NextGen Institute · Machine Learning Final Project**  
*Presenter: Machine Learning Engineer & Data Scientist*  
*Target Audience: E-Commerce Growth Directors, Head of CRO, Lead Data Scientists*

---

### Slide 1: Title & Overview
* **Title:** AI-Driven Purchase Intent Prediction & Dynamic Intervention Engine
* **Subtitle:** Maximizing E-Commerce Revenue and Eliminating Coupon Cannibalization with Real-Time ML
* **Key Stats:** Trained on 12,330 sessions · 84.03% Buyer Recall · 0.9329 Test ROC-AUC · Built with Scikit-Learn, LightGBM & Streamlit.

---

### Slide 2: The Business Problem & Opportunity
* **The Challenge:** Over 84% of website visitors abandon e-commerce sessions without purchasing.
* **The Cost of Blind Discounting:**
  - Giving coupons to everyone drains profit margins (cannibalization).
  - Giving discounts to non-serious visitors wastes marketing budget.
* **The Solution:** A machine learning model predicting buyer conversion probability in real-time (<5ms) during the active browsing session, triggering targeted automated interventions.

---

### Slide 3: Data Understanding & Target Definition
* **Dataset:** Online Shoppers Purchasing Intention (UCI ML Repository, Sakar et al.).
* **Scale:** 12,330 browsing sessions across 10 calendar months.
* **Target:** `Revenue` (1 = Completed Purchase, 0 = Session Abandoned).
* **Imbalance Reality:** 84.5% non-buyers vs 15.5% buyers (5.46:1 imbalance ratio).
* **Core Insight:** Accuracy is a misleading metric; an empty model predicting "No Purchase" achieves 84.5% accuracy but zero business value.

---

### Slide 4: Key EDA Insights (Data Signals)
* **Insight 1: PageValues Dominance:** 78.4% of converted shoppers visited pages with positive Google Analytics `PageValues` (average score: 27.2 vs 1.9 for abandoners).
* **Insight 2: Exit & Bounce Rates:** Non-converting visitors have 3x higher bounce rates and 2.5x higher exit rates.
* **Insight 3: High Seasonality (November):** November captures the highest volume and a 25.3% conversion rate, driven by Black Friday and holiday gift intent.
* **Insight 4: Returning vs. New Visitors:** New visitors convert at higher rates when product engagement is deep, while returning visitors represent 85%+ of total sessions.

---

### Slide 5: Data Preprocessing & Zero-Leakage Architecture
* **Strict Stratified Split (80/20):**
  - Train: 9,864 sessions | Test: 2,466 sessions (held untouched until final evaluation).
  - All scalers, imputers, and encoders fit strictly on training splits.
* **Domain Feature Engineering:**
  - `Total_Pages` & `Total_Duration`: Aggregate session depth.
  - `Product_Page_Ratio`: Percentage of activity focused on merchandise vs administrative/account pages.
  - `PageValue_Bounce_Ratio`: High-intent signal scaling page value against bounce friction.
* **Scikit-Learn Pipeline:** Encapsulated inside `ColumnTransformer` with `StandardScaler` and `OneHotEncoder(handle_unknown='ignore')`.

---

### Slide 6: Modeling Strategy & Baselines
* **Algorithm Comparison:**
  1. **Dummy Baseline:** Majority class predictor (Accuracy: 84.5%, Recall: 0.0%, PR-AUC: 0.155).
  2. **Logistic Regression (Balanced):** Scaled linear baseline (ROC-AUC: 0.9058, Recall: 76.1%).
  3. **Random Forest (Balanced):** 150-tree bagging ensemble (ROC-AUC: 0.9300, Recall: 82.3%).
  4. **LightGBM (Gradient Boosting):** Tuned GBDT handling scale-positive weight (ROC-AUC: 0.9319, PR-AUC: 0.7458).

---

### Slide 7: Hyperparameter Tuning (Training Set Only)
* **Strategy:** 5-Fold Stratified Cross-Validation on training data.
* **Optimization Objective:** `average_precision` (PR-AUC) to prioritize rare-class conversion.
* **Best Hyperparameters Found:**
  - `learning_rate`: 0.03
  - `n_estimators`: 200
  - `num_leaves`: 15
  - `min_child_samples`: 50
  - `scale_pos_weight`: 5.46

---

### Slide 8: Final Test Set Results & Benchmark
* **Tested on untouched 2,466 held-out sessions:**
  - **Test ROC-AUC:** **0.9329**
  - **Test PR-AUC:** **0.7391** (vs. 0.1547 baseline!)
  - **Buyer Recall:** **84.03%** (Detected 321 of 382 actual buyers)
  - **False Negatives:** Only 61 missed buyers across entire test set.
  - **Overall Accuracy:** 85.93% (achieved while maintaining high sensitivity).

---

### Slide 9: Error Analysis & Key Drivers
* **Top Predictive Features:**
  1. `ExitRates` (Inversely correlated with buying)
  2. `Product_Page_Ratio` (Engineered)
  3. `PageValue_Bounce_Ratio` (Engineered)
  4. `PageValues`
  5. `ProductRelated_Duration`
* **Error Analysis Findings:**
  - *False Positives (286 sessions):* Visitors with moderate PageValues who added items to carts but abandoned due to external checkout friction (shipping costs, payment gate).
  - *False Negatives (61 sessions):* Extremely fast buyers (impulse purchases) who bought with very short duration and few pageviews.

---

### Slide 10: Business Recommendation & Decision Policy
* **Tier 1 — High Purchase Intent ($P \ge 0.70$):**
  - *Action:* **Protect Margin (Zero Discount)**. Provide 1-click checkout and trust guarantees.
* **Tier 2 — Borderline / Hesitant Intent ($0.35 \le P < 0.70$):**
  - *Action:* **Intervene in Real Time**. Trigger an exit-intent popup with a **15% discount voucher** or initiate proactive live chat assistance.
* **Tier 3 — Casual Browsing ($P < 0.35$):**
  - *Action:* **Passive Navigation**. No discounts; collect email newsletter signups.

---

### Slide 11: Strategic Campaign ROI Simulation (The Stretch Goal)
* **Monthly Scenario: 50,000 Visitors, $75 AOV, 35% Gross Margin:**
  - Targeting the 15% hesitant segment (7,500 visitors) with an exit-intent 15% voucher:
  - Extra orders won: **+1,500 orders/month**.
  - Gross incremental revenue: **+$112,500/month**.
  - Net incremental profit after coupon discount: **+$22,500/month**.
  - **Return on Investment (ROI):** High positive cash return while saving organic revenue!

---

### Slide 12: Production Web App & Next Steps
* **Production Streamlit App (`app.py`):**
  - Real-time visitor session simulator.
  - Automated dynamic prescription engine (No Discount vs Coupon vs Chat).
  - Interactive What-If ROI Calculator for marketing teams.
* **Next Steps:**
  - A/B testing intervention policy against status-quo blanket discounting.
  - Integrating real-time WebSocket events from Shopify / Magento / custom front-ends.
