"""
Streamlit Web Application: Real-Time E-Commerce Purchase Intent & Dynamic Intervention Engine
NextGen Institute · Final Project Deliverable
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set page layout configuration
st.set_page_config(
    page_title="Purchase Intent AI | E-Commerce Conversion",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .recommendation-box {
        padding: 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 15px;
    }
    .high-intent {
        background-color: #DEF7EC;
        color: #03543F;
        border-left: 5px solid #0E9F6E;
    }
    .medium-intent {
        background-color: #FEF08A;
        color: #713F12;
        border-left: 5px solid #EAB308;
    }
    .low-intent {
        background-color: #F3F4F6;
        color: #374151;
        border-left: 5px solid #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOAD MODEL & PIPELINE
# ------------------------------------------------------------------
@st.cache_resource
def load_model_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "best_pipeline.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}. Please run train_pipeline.py first.")
        st.stop()
    return joblib.load(model_path)

model_pipeline = load_model_pipeline()

# ------------------------------------------------------------------
# HELPER: FEATURE ENGINEERING
# ------------------------------------------------------------------
def prepare_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df['Total_Pages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['Product_Page_Ratio'] = df['ProductRelated'] / (df['Total_Pages'] + 1e-5)
    df['Avg_Duration_Per_Page'] = df['Total_Duration'] / (df['Total_Pages'] + 1e-5)
    df['PageValue_Bounce_Ratio'] = df['PageValues'] / (df['BounceRates'] + 1e-4)
    
    nominal_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
    for col in nominal_cols:
        df[col] = df[col].astype(str)
        
    return df

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown('<div class="main-header">🛒 Real-Time Purchase Intent & Conversion Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict online session conversion in real-time and recommend smart business interventions (coupons, chat assist, or margin protection).</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Real-Time Visitor Prediction",
    "📈 Campaign ROI Simulator (Stretch Goal)",
    "📊 Model Performance & Explainability"
])

# ------------------------------------------------------------------
# TAB 1: VISITOR PREDICTION FORM
# ------------------------------------------------------------------
with tab1:
    st.subheader("Simulate a Live Browsing Session")
    
    col_preset, _ = st.columns([2, 1])
    with col_preset:
        preset = st.radio(
            "Quick Demo Personas:",
            ["Custom Input", "🟢 Hot Buyer (High PageValues, deep product browsing)", "🟡 Hesitant Abandoner (Bouncing, moderate interest)", "⚪ Casual Window Shopper (Brief bounce)"],
            horizontal=True
        )

    # Defaults based on preset
    if preset == "🟢 Hot Buyer (High PageValues, deep product browsing)":
        def_admin, def_admin_dur = 3, 95.0
        def_info, def_info_dur = 2, 45.0
        def_prod, def_prod_dur = 28, 1150.0
        def_bounce, def_exit = 0.002, 0.015
        def_pageval = 32.5
        def_visitor = "Returning_Visitor"
        def_month = "Nov"
    elif preset == "🟡 Hesitant Abandoner (Bouncing, moderate interest)":
        def_admin, def_admin_dur = 1, 20.0
        def_info, def_info_dur = 0, 0.0
        def_prod, def_prod_dur = 12, 320.0
        def_bounce, def_exit = 0.025, 0.055
        def_pageval = 6.8
        def_visitor = "Returning_Visitor"
        def_month = "Dec"
    elif preset == "⚪ Casual Window Shopper (Brief bounce)":
        def_admin, def_admin_dur = 0, 0.0
        def_info, def_info_dur = 0, 0.0
        def_prod, def_prod_dur = 3, 25.0
        def_bounce, def_exit = 0.120, 0.150
        def_pageval = 0.0
        def_visitor = "New_Visitor"
        def_month = "May"
    else:
        def_admin, def_admin_dur = 1, 30.0
        def_info, def_info_dur = 0, 0.0
        def_prod, def_prod_dur = 15, 450.0
        def_bounce, def_exit = 0.01, 0.03
        def_pageval = 8.0
        def_visitor = "Returning_Visitor"
        def_month = "Nov"

    with st.form("visitor_session_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📄 Page Browsing Activity")
            admin_pages = st.number_input("Administrative Pages Visited", min_value=0, max_value=50, value=def_admin)
            admin_dur = st.number_input("Administrative Duration (seconds)", min_value=0.0, max_value=3600.0, value=float(def_admin_dur))
            info_pages = st.number_input("Informational Pages Visited", min_value=0, max_value=50, value=def_info)
            info_dur = st.number_input("Informational Duration (seconds)", min_value=0.0, max_value=3600.0, value=float(def_info_dur))
            prod_pages = st.number_input("Product Related Pages Visited", min_value=0, max_value=300, value=def_prod)
            prod_dur = st.number_input("Product Related Duration (seconds)", min_value=0.0, max_value=15000.0, value=float(def_prod_dur))

        with col2:
            st.markdown("##### 🔍 Engagement & Google Analytics")
            page_values = st.number_input("PageValues (Average Value Score)", min_value=0.0, max_value=400.0, value=float(def_pageval), 
                                         help="The average value for a web page that a user visited before completing an e-commerce transaction.")
            bounce_rate = st.slider("Bounce Rate", min_value=0.0, max_value=0.20, value=float(def_bounce), step=0.005,
                                    help="Percentage of visitors who enter the site and leave immediately.")
            exit_rate = st.slider("Exit Rate", min_value=0.0, max_value=0.20, value=float(def_exit), step=0.005,
                                  help="Percentage of pageviews that were the last in the session.")
            special_day = st.slider("Special Day Closeness (Holidays/Events)", min_value=0.0, max_value=1.0, value=0.0, step=0.2)

        with col3:
            st.markdown("##### 👤 Session Context & Visitor Type")
            month = st.selectbox("Month of Session", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].index(def_month))
            visitor_type = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'], index=['Returning_Visitor', 'New_Visitor', 'Other'].index(def_visitor))
            weekend = st.checkbox("Session on Weekend?", value=False)
            traffic_type = st.selectbox("Traffic Channel ID", [1, 2, 3, 4, 5, 8, 10, 13, 20], index=1)
            os_type = st.selectbox("Operating System ID", [1, 2, 3, 4, 8], index=1)
            browser_type = st.selectbox("Browser ID", [1, 2, 4, 6, 8], index=1)
            region_type = st.selectbox("Region ID", [1, 2, 3, 4, 7, 9], index=0)

        submitted = st.form_submit_button("⚡ Predict Purchase Intent & Prescribe Action", use_container_width=True)

    if submitted or preset != "Custom Input":
        input_data = pd.DataFrame([{
            'Administrative': admin_pages,
            'Administrative_Duration': admin_dur,
            'Informational': info_pages,
            'Informational_Duration': info_dur,
            'ProductRelated': prod_pages,
            'ProductRelated_Duration': prod_dur,
            'BounceRates': bounce_rate,
            'ExitRates': exit_rate,
            'PageValues': page_values,
            'SpecialDay': special_day,
            'Month': month,
            'OperatingSystems': str(os_type),
            'Browser': str(browser_type),
            'Region': str(region_type),
            'TrafficType': str(traffic_type),
            'VisitorType': visitor_type,
            'Weekend': weekend
        }])

        prepared_input = prepare_input(input_data)
        prob = model_pipeline.predict_proba(prepared_input)[0, 1]
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Result & Automated Intervention Policy")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Predicted Purchase Probability", value=f"{prob * 100:.1f}%")
            st.progress(float(prob))
            
            if prob >= 0.70:
                intent_label = "🔥 High Purchase Intent"
                badge_color = "normal"
            elif prob >= 0.35:
                intent_label = "⚡ Borderline / Hesitant Intent"
                badge_color = "warning"
            else:
                intent_label = "❄️ Low Browsing Intent"
                badge_color = "off"
            st.write(f"**Customer Intent Segment:** {intent_label}")

        with res_col2:
            if prob >= 0.70:
                st.markdown("""
                <div class="recommendation-box high-intent">
                    ✅ <b>PRESCRIPTION: PROTECT MARGIN (NO DISCOUNT)</b><br>
                    <span style="font-size:0.95rem; font-weight:normal;">
                    This customer demonstrates clear, active buying signals. Giving a discount coupon here would cause <b>margin cannibalization</b>. 
                    <b>Action:</b> Seamless checkout experience, fast shipping badges, cross-sell accessories at full price.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            elif prob >= 0.35:
                st.markdown("""
                <div class="recommendation-box medium-intent">
                    🎁 <b>PRESCRIPTION: INTERVENE TO CLOSE DEAL (TRIGGER 15% COUPON / LIVE CHAT)</b><br>
                    <span style="font-size:0.95rem; font-weight:normal;">
                    This user has significant purchase interest but is hesitating or at risk of leaving the cart.
                    <b>Action:</b> Trigger an exit-intent popup with a <b>15% limited-time discount</b> or prompt customer support live chat: <i>"Can we help answer questions about your selected items?"</i>
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation-box low-intent">
                    👀 <b>PRESCRIPTION: PASSIVE BROWSING (DO NOT DISCOUNT)</b><br>
                    <span style="font-size:0.95rem; font-weight:normal;">
                    Visitor is casually exploring or bouncing quickly. Offering discounts will yield low conversion while draining promotional budgets.
                    <b>Action:</b> Display popular category highlights or newsletter subscription banner without active cart discounting.
                    </span>
                </div>
                """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# TAB 2: CAMPAIGN ROI SIMULATOR (STRETCH GOAL)
# ------------------------------------------------------------------
with tab2:
    st.subheader("💡 Strategic Campaign Simulator (Stretch Goal)")
    st.markdown("""
    Evaluate the financial return of implementing an automated intervention policy on your e-commerce store.
    Simulate offering targeted exit-intent coupons to borderline/hesitant visitors vs. no intervention.
    """)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.markdown("##### Campaign Parameters")
        monthly_sessions = st.number_input("Monthly Browsing Sessions", min_value=1000, max_value=1000000, value=50000, step=5000)
        avg_order_value = st.number_input("Average Order Value (AOV) [$]", min_value=10.0, max_value=500.0, value=75.0, step=5.0)
        profit_margin = st.slider("Product Gross Profit Margin (%)", min_value=10, max_value=80, value=35, step=5) / 100.0
        
    with sim_col2:
        st.markdown("##### Intervention Strategy")
        borderline_pct = st.slider("Targeted Hesitant Visitors (Share of Sessions)", min_value=0.05, max_value=0.30, value=0.15, step=0.01,
                                   help="Percentage of sessions classified into the 35% - 70% hesitant intent bucket.")
        coupon_discount = st.slider("Offered Discount Rate (%)", min_value=5, max_value=30, value=15, step=1) / 100.0
        conversion_uplift = st.slider("Estimated Conversion Uplift among Hesitant (%)", min_value=5, max_value=40, value=20, step=1) / 100.0

    # Economics calculation
    targeted_sessions = monthly_sessions * borderline_pct
    baseline_conversions = targeted_sessions * 0.10 # without coupon
    new_conversions = targeted_sessions * (0.10 + conversion_uplift)
    incremental_orders = new_conversions - baseline_conversions
    
    # Financial metrics
    gross_incremental_revenue = incremental_orders * avg_order_value
    discount_cost = new_conversions * (avg_order_value * coupon_discount)
    gross_margin_earned = gross_incremental_revenue * profit_margin
    net_incremental_profit = gross_margin_earned - discount_cost
    roi = (net_incremental_profit / discount_cost * 100) if discount_cost > 0 else 0

    st.markdown("---")
    st.markdown("#### Projected Monthly Financial Impact")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Targeted Sessions", f"{int(targeted_sessions):,}")
    m2.metric("Extra Orders Won", f"+{int(incremental_orders):,}")
    m3.metric("Incremental Revenue", f"${gross_incremental_revenue:,.0f}")
    m4.metric("Net Incremental Profit", f"${net_incremental_profit:,.0f}", delta=f"{roi:.1f}% ROI")

    st.info(f"""
    **Executive Summary:**
    By targeting **{int(targeted_sessions):,} hesitant sessions** with personalized interventions instead of mass-discounting:
    - You capture **{int(incremental_orders):,} additional customer orders** each month.
    - Total new revenue generated: **${gross_incremental_revenue:,.0f}**.
    - Net added profit after covering coupon discounts: **${net_incremental_profit:,.0f}**.
    - Prevents wasting discounts on the organic 85%+ intent buyers!
    """)

# ------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & EXPLAINABILITY
# ------------------------------------------------------------------
with tab3:
    st.subheader("Model Architecture & Benchmarking")
    st.markdown("""
    Evaluation results conducted on an **untouched 20% test set (2,466 sessions)** using strict time-aware and stratified splitting.
    """)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Test ROC-AUC", "0.9329", "+0.4329 vs Baseline")
    b2.metric("Test PR-AUC", "0.7391", "+0.5844 vs Baseline")
    b3.metric("Buyer Recall", "84.03%", "Catches 321 of 382 buyers")
    b4.metric("Total Accuracy", "85.93%", "Balanced Precision/Recall")

    st.markdown("#### Model Comparison (5-Fold Stratified Cross-Validation on Training Data)")
    comparison_data = pd.DataFrame({
        "Model": ["Baseline (Majority Class)", "Logistic Regression (Balanced)", "Random Forest (Balanced)", "Tuned LightGBM (Champion)"],
        "ROC-AUC": [0.5000, 0.9058, 0.9300, 0.9329],
        "PR-AUC": [0.1547, 0.6680, 0.7361, 0.7458],
        "Recall": ["0.0%", "76.1%", "82.3%", "84.0%"],
        "Business Actionability": ["Useless (predicts all False)", "Good baseline, linear limit", "High accuracy, slower inference", "Optimal, ultra-fast real-time inference (<5ms)"]
    })
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        cm_path = os.path.join(fig_dir, "eval_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix on Untouched Test Set (2,466 sessions)")
            
    with col_f2:
        fi_path = os.path.join(fig_dir, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Top Most Influential Session Features")

    st.markdown("""
    **Key Takeaways from Feature Importance:**
    1. **ExitRates & PageValues:** Strongest predictors of intent. High PageValues indicate user looked at high-converting pages.
    2. **Product_Page_Ratio (Engineered):** Users with a high percentage of product pages vs informational pages are far more likely to buy.
    3. **PageValue_Bounce_Ratio (Engineered):** High page value combined with low bounce indicates decisive intent.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.85rem;'>"
    "NextGen Institute · Machine Learning Final Project · Developed with Scikit-Learn, LightGBM & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
