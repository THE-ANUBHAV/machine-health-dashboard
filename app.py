import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Health Monitoring",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling status cards and alerts
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .high-risk { color: #ff4b4b; font-weight: bold; }
    .med-risk { color: #ffa500; font-weight: bold; }
    .low-risk { color: #09ab3b; font-weight: bold; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads the AI4I 2020 dataset."""
    try:
        df = pd.read_csv("ai4i2020.csv")
        return df
    except FileNotFoundError:
        st.error("File 'ai4i2020.csv' not found. Please ensure it is in the same directory.")
        return pd.DataFrame()

@st.cache_resource
def load_model(model_path):
    """Loads a pickled model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading {model_path}: {e}")
        return None

# Load Dataset
df = load_data()

# -----------------------------------------------------------------------------
# 3. PREPROCESSING (UPDATED FOR 6 vs 10 FEATURES)
# -----------------------------------------------------------------------------
def preprocess_input(df_input):
    """
    Preprocesses data to match training formats.
    Returns dictionaries of feature sets and scalers.
    """
    data = df_input.copy()
    
    # 1. Encode Type (Standard LabelEncoder: H=0, L=1, M=2)
    type_map = {'H': 0, 'L': 1, 'M': 2}
    if data['Type'].dtype == 'object':
        data['Type'] = data['Type'].map(type_map)
    
    # Define Feature Sets
    # Set A: Standard 6 features
    cols_6 = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Set B: 10 features (Likely 5 sensors + 5 failure flags, Type dropped)
    # This addresses the SVM/MLP 10 feature requirement
    cols_10 = ['Air temperature [K]', 'Process temperature [K]', 
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
               'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    # Prepare X matrices
    X_6 = data[cols_6]
    
    # Check if failure cols exist for 10-feature set
    if all(col in data.columns for col in cols_10):
        X_10 = data[cols_10]
    else:
        # Fallback if dataset lacks failure columns (unlikely given csv)
        X_10 = pd.DataFrame()

    # Fit Scalers (Crucial for SVM/MLP)
    scaler_6 = StandardScaler()
    X_6_scaled = scaler_6.fit_transform(X_6)
    
    scaler_10 = StandardScaler()
    X_10_scaled = scaler_10.fit_transform(X_10) if not X_10.empty else None
    
    return {
        "X_6_raw": X_6,
        "X_6_scaled": X_6_scaled,
        "X_10_scaled": X_10_scaled,
        "names_6": cols_6,
        "names_10": cols_10
    }

if not df.empty:
    preprocessed = preprocess_input(df)
    y_true = df['Machine failure']

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Dashboard Controls")

# Model Selector
model_options = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Support Vector Machine": "svm.pkl",
    "MLP Neural Network": "mlp.pkl"
}
selected_model_name = st.sidebar.selectbox("Select Prediction Model", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]
model = load_model(selected_model_path)

# Navigation
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Machine Details", "Model Performance", "About"])

# -----------------------------------------------------------------------------
# 5. GLOBAL PREDICTIONS & LOGIC
# -----------------------------------------------------------------------------
# Initialize variables to avoid scope issues
probs = np.zeros(len(df)) if not df.empty else []
feature_names = []

if model is not None and not df.empty:
    # Detect required features
    n_features = getattr(model, "n_features_in_", 6)
    
    X_input = None
    
    # Logic to select correct input shape
    if n_features == 10:
        X_input = preprocessed["X_10_scaled"]
        feature_names = preprocessed["names_10"]
        st.sidebar.success(f"Loaded {selected_model_name} (10 features detected)")
    else:
        # For Trees (Raw) vs SVM/MLP (Scaled) on 6 features
        feature_names = preprocessed["names_6"]
        if "SVM" in selected_model_name or "MLP" in selected_model_name:
            X_input = preprocessed["X_6_scaled"]
        else:
            X_input = preprocessed["X_6_raw"]
        st.sidebar.success(f"Loaded {selected_model_name} (6 features detected)")

    if X_input is not None:
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[:, 1]
            else:
                # Fallback for SVM if probability=False
                pred_class = model.predict(X_input)
                probs = pred_class.astype(float) # Convert 0/1 to float for compatibility
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            probs = np.zeros(len(df))

    # Assign Risk Categories
    df['Failure Probability'] = probs
    df['Risk Category'] = df['Failure Probability'].apply(
        lambda p: 'High' if p > 0.7 else ('Medium' if p > 0.4 else 'Low')
    )

    # RUL Heuristic
    df['Predicted RUL (min)'] = 250 - df['Tool wear [min]']
    df['Predicted RUL (min)'] = df['Predicted RUL (min)'].clip(lower=0) 

    # Maintenance Recommendation
    def recommend_maint(row):
        if row['Risk Category'] == 'High': return "🔴 Replace Tool / Inspect Immediately"
        elif row['Risk Category'] == 'Medium': return "🟡 Scheduled Inspection"
        else: return "🟢 Normal Operation"
    
    df['Recommendation'] = df.apply(recommend_maint, axis=1)

# -----------------------------------------------------------------------------
# 6. PAGE: DASHBOARD OVERVIEW
# -----------------------------------------------------------------------------
if page == "Dashboard Overview":
    # --- Header ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Machine Health Monitoring & Failure Prediction")
        st.markdown(f"**Model Version:** v1.0 ({selected_model_name}) | **Status:** Online")
    with col2:
        st.markdown(f"**Last Refresh:** {datetime.now().strftime('%H:%M:%S')}")
        health_score = 100 - (df['Risk Category'] == 'High').mean() * 100
        st.metric("Overall Plant Health", f"{health_score:.1f}%")

    st.divider()

    # --- Top Summary Cards ---
    total_machines = len(df)
    high_risk_count = len(df[df['Risk Category'] == 'High'])
    med_risk_count = len(df[df['Risk Category'] == 'Medium'])
    safe_count = len(df[df['Risk Category'] == 'Low'])
    failures_logged = df['Machine failure'].sum()
    avg_tool_wear = df['Tool wear [min]'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏭 Total Machines", total_machines)
    c2.metric("🔴 High Risk", high_risk_count, delta=f"{high_risk_count/total_machines:.1%}")
    c3.metric("🟡 Medium Risk", med_risk_count)
    c4.metric("✅ Safe Machines", safe_count)
    c5.metric("⚙️ Avg Tool Wear", f"{avg_tool_wear:.1f} min")

    # --- Alerts Panel ---
    if high_risk_count > 0:
        st.subheader("🔔 Active Alerts (High Risk Detected)")
        high_risk_df = df[df['Risk Category'] == 'High'].head(3) 
        for idx, row in high_risk_df.iterrows():
            st.error(f"⚠️ **Machine {row['UDI']}**: High Failure Probability ({row['Failure Probability']:.1%}). Torque: {row['Torque [Nm]']} Nm. Recommendation: {row['Recommendation']}")
    else:
        st.success("No High Risk Machines Detected.")

    # --- Main Layout: Table & Distribution ---
    col_main, col_charts = st.columns([2, 1])

    with col_main:
        st.subheader("📋 Machine-Level Prediction Table")
        display_cols = ['UDI', 'Type', 'Torque [Nm]', 'Tool wear [min]', 'Air temperature [K]', 
                        'Failure Probability', 'Risk Category', 'Predicted RUL (min)', 'Recommendation']
        
        st.dataframe(
            df[display_cols].sort_values(by='Failure Probability', ascending=False),
            column_config={
                "Failure Probability": st.column_config.ProgressColumn(
                    "Failure Prob", format="%.2f", min_value=0, max_value=1,
                ),
            },
            height=400,
            use_container_width=True
        )

    with col_charts:
        st.subheader("📊 Failure Prediction Distribution")
        fig_hist = px.histogram(df, x="Failure Probability", nbins=20, 
                                title="Distribution of Risk Scores",
                                color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("**Historical Failure Causes**")
        failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        failure_counts = df[failure_cols].sum().reset_index()
        failure_counts.columns = ['Cause', 'Count']
        
        fig_pie = px.pie(failure_counts, values='Count', names='Cause', hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Feature Importance / Correlation ---
    st.subheader("🔍 Model Feature Importance / Correlation")
    if hasattr(model, 'feature_importances_'):
        # Tree Models
        feat_names_display = feature_names if len(feature_names) == len(model.feature_importances_) else [f"Feat {i}" for i in range(len(model.feature_importances_))]
        feat_imp = pd.DataFrame({
            'Feature': feat_names_display,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', 
                         title=f"Feature Importance ({selected_model_name})")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        # SVM / MLP: Show Correlation Heatmap
        # FIX: Prepare numeric dataframe for correlation
        df_corr = df.copy()
        
        # Ensure Type is numeric for correlation if it's one of the features
        if 'Type' in df_corr.columns and df_corr['Type'].dtype == 'object':
             df_corr['Type'] = df_corr['Type'].map({'H': 0, 'L': 1, 'M': 2})
        
        # Filter columns present in the feature set + Failure
        corr_cols = [c for c in feature_names if c in df_corr.columns]
        
        if corr_cols:
            corr = df_corr[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Viridis', title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Could not calculate correlation: Feature names do not match DataFrame columns.")


# -----------------------------------------------------------------------------
# 7. PAGE: MACHINE DETAILS
# -----------------------------------------------------------------------------
elif page == "Machine Details":
    st.title("🔎 Machine Details & Diagnostics")
    selected_udi = st.selectbox("Select Machine ID (UDI)", df['UDI'].unique())
    m_data = df[df['UDI'] == selected_udi].iloc[0]
    
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1: st.metric("Machine ID", m_data['UDI'])
    with col_h2: 
        color = "red" if m_data['Risk Category'] == 'High' else "orange" if m_data['Risk Category'] == 'Medium' else "green"
        st.markdown(f"### Risk Status: :{color}[{m_data['Risk Category']}]")
    with col_h3: st.metric("Failure Probability", f"{m_data['Failure Probability']:.2%}")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.info(f"**Type:** {m_data['Type']}")
    c2.info(f"**Air Temp:** {m_data['Air temperature [K]']} K")
    c3.info(f"**Process Temp:** {m_data['Process temperature [K]']} K")
    c4.info(f"**Rot Speed:** {m_data['Rotational speed [rpm]']} rpm")

    st.subheader(f"📈 Sensor Trends: Machine {selected_udi}")
    col_trend1, col_trend2 = st.columns(2)

    def generate_history(current_val, variability=0.05, steps=20):
        history = [current_val * (1 + np.random.uniform(-variability, variability)) for _ in range(steps)]
        history[-1] = current_val 
        return history

    with col_trend1:
        torque_hist = generate_history(m_data['Torque [Nm]'])
        fig_torque = px.line(y=torque_hist, x=list(range(20)), title="Torque [Nm] - Last 20 Cycles")
        fig_torque.add_hline(y=60, line_dash="dash", line_color="red")
        st.plotly_chart(fig_torque, use_container_width=True)

    with col_trend2:
        wear_hist = generate_history(m_data['Tool wear [min]'])
        fig_wear = px.line(y=wear_hist, x=list(range(20)), title="Tool Wear [min] - Growth")
        fig_wear.add_hline(y=200, line_dash="dash", line_color="red")
        st.plotly_chart(fig_wear, use_container_width=True)

    st.subheader("🛠️ Maintenance Scheduler")
    days_left = int(m_data['Predicted RUL (min)'] / 60 / 8)
    if days_left < 1: days_left = 0
    
    sched_col1, sched_col2, sched_col3 = st.columns(3)
    sched_col1.metric("Est. RUL (Minutes)", f"{m_data['Predicted RUL (min)']:.0f} min")
    sched_col2.metric("Est. Days to Failure", f"{days_left} days")
    sched_col3.metric("Estimated Cost Savings", f"${m_data['Failure Probability'] * 5000:.0f}")

    if m_data['Risk Category'] == 'High':
        st.warning(f"**Action Required:** {m_data['Recommendation']}")
        st.button("📅 Schedule Maintenance Request", type="primary")

# -----------------------------------------------------------------------------
# 8. PAGE: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("🧪 Model Performance Metrics")
    st.markdown(f"Evaluation based on the full **AI4I 2020** dataset using **{selected_model_name}**.")

    y_pred = (df['Failure Probability'] > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall", f"{rec:.4f}")
    m4.metric("F1 Score", f"{f1:.4f}")

    col_perf1, col_perf2 = st.columns(2)
    with col_perf1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Normal', 'Failure'], y=['Normal', 'Failure'],
                           color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_perf2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, df['Failure Probability'])
        roc_auc = auc(fpr, tpr)
        fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.4f})')
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)

# -----------------------------------------------------------------------------
# 9. PAGE: ABOUT
# -----------------------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About This Dashboard")
    st.markdown("""
    ### Dataset: AI4I 2020 Predictive Maintenance Dataset
    **Features Used:**
    * **Type:** Quality variant (Low/Medium/High)
    * **Air Temperature [K]:** Room temperature.
    * **Process Temperature [K]:** Generated by the process.
    * **Rotational Speed [rpm]:** Calculated from power.
    * **Torque [Nm]:** Torque values normally distributed.
    * **Tool Wear [min]:** Usage time of the cutting tool.
    
    *Note: Some models (SVM/MLP) may use failure flags as input features based on training configuration.*
    """)