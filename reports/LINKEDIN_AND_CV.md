# Portfolio Deliverables: CV Bullet & LinkedIn Post
**NextGen Institute · Machine Learning Final Project**

---

## 1. Resume / CV Bullet (Portfolio-Ready)

### Primary Version:
> **Developed a real-time purchase-intent machine learning engine on 12,330 e-commerce sessions to detect customer conversion and prescribe margin-protecting interventions, achieving 0.933 ROC-AUC, 0.739 PR-AUC, and 84.0% buyer recall using LightGBM, Scikit-Learn pipelines, and Streamlit.**

### Condensed Version:
> **Built an e-commerce purchase-intent predictor across 12K+ browsing sessions with leakage-free Scikit-Learn pipelines and tuned LightGBM, reaching 0.933 ROC-AUC (84.0% buyer recall) and deployed a real-time intervention and ROI simulation web app on Streamlit.**

---

## 2. Professional LinkedIn Post

**Post Text (Ready to Publish):**

Most e-commerce websites face the same painful reality: over 84% of visitors browse and leave without buying a single item.

To fix this, many brands make an expensive mistake — giving discount coupons to everyone. That either wastes money on shoppers who would have bought anyway (margin cannibalization) or gives discounts to casual browsers who won’t buy regardless.

In my NextGen Institute Machine Learning internship, our team engineered an end-to-end Machine Learning system that predicts buyer intent in real-time during an active session — and prescribes the right business action before the visitor leaves:

🎯 **Key Results & Engineering Highlights:**
• Trained, tuned, and evaluated models across 12,330 real browsing sessions using strict leakage-free Scikit-Learn pipelines.  
• Outperformed linear and tree baselines with a tuned LightGBM model, achieving a **0.9329 Test ROC-AUC**, **0.7391 PR-AUC**, and **84.03% Buyer Recall** (detecting 321 of 382 actual purchasers on untouched test data).  
• Engineered custom behavioral intent metrics (e.g. `Product_Page_Ratio`, `PageValue_Bounce_Ratio`) that proved to be top drivers alongside Google Analytics `PageValues` and `ExitRates`.  
• Built a dynamic business decision policy:
   - High Intent (>70%): Protect margins (0% discount).
   - Borderline/Hesitant (35%-70%): Trigger timed 15% exit-intent voucher or live chat.
   - Low Intent (<35%): Passive browsing (no discount).  
• Shipped a live, interactive **Streamlit Web Application** featuring a real-time visitor simulator and a Campaign ROI What-If Calculator.

💡 **The Biggest Takeaway:** In severe class imbalance (~15.5% conversion), **Accuracy is deceptive** (a dummy model gets 84.5% accuracy with 0% utility). Optimizing directly for PR-AUC and Recall is what unlocks real commercial value.

🔗 **Try the live app:** [Streamlit App Link]  
💻 **Code & Notebook:** https://github.com/patrickemad20112007-cpu/ecommerce-purchase-intent  

Huge thanks to our instructors and mentors at NextGen Institute!

#MachineLearning #DataScience #Ecommerce #LightGBM #Python #Streamlit #GrowthMarketing #ConversionRateOptimization
