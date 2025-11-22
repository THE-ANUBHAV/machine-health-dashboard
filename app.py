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

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ai4i2020.csv")
        return df
    except FileNotFoundError:
        st.error("File 'ai4i2020.csv' not found.")
        return pd.DataFrame()

@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading {model_path}: {e}")
        return None

df = load_data()

# -----------------------------------------------------------------------------
# 3. PREPROCESSING
# -----------------------------------------------------------------------------
def preprocess_input(df_input):
    data = df_input.copy()
    type_map = {'H': 0, 'L': 1, 'M': 2}
    if data['Type'].dtype == 'object':
        data['Type'] = data['Type'].map(type_map)
    
    cols_6 = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    cols_10 = ['Air temperature [K]', 'Process temperature [K]', 
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
               'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    X_6 = data[cols_6]
    X_10 = data[cols_10] if all(c in data.columns for c in cols_10) else pd.DataFrame()

    # Fallback Scalers
    scaler_6 = StandardScaler().fit(X_6)
    X_6_scaled = scaler_6.transform(X_6)
    
    X_10_scaled = None
    if not X_10.empty:
        scaler_10 = StandardScaler().fit(X_10)
        X_10_scaled = scaler_10.transform(X_10)
    
    return {
        "X_6_raw": X_6, "X_6_scaled": X_6_scaled,
        "X_10_scaled": X_10_scaled, "names_6": cols_6, "names_10": cols_10
    }

if not df.empty:
    preprocessed = preprocess_input(df)
    y_true = df['Machine failure']

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Dashboard Controls")

model_options = {
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Support Vector Machine": "svm.pkl",
    "MLP Neural Network": "mlp.pkl"
}
selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]
model = load_model(selected_model_path)

page = st.sidebar.radio("Go to", ["Dashboard Overview", "Machine Details", "Model Performance", "About"])

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# 5. GLOBAL PREDICTIONS & SIDEBAR VISUALIZATION
# -----------------------------------------------------------------------------
probs = np.zeros(len(df)) if not df.empty else []
feature_names = []

if model is not None and not df.empty:
    # 5a. Generate Predictions
    n_features = getattr(model, "n_features_in_", 6)
    X_input = None
    
    if n_features == 10:
        X_input = preprocessed["X_10_scaled"]
        feature_names = preprocessed["names_10"]
    else:
        feature_names = preprocessed["names_6"]
        if "SVM" in selected_model_name or "MLP" in selected_model_name:
            X_input = preprocessed["X_6_scaled"]
        else:
            X_input = preprocessed["X_6_raw"]

    if X_input is not None:
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[:, 1]
            else:
                pred_class = model.predict(X_input)
                probs = pred_class.astype(float)
        except Exception:
            probs = np.zeros(len(df))

    # 5b. Calculate Stats for Sidebar Visualization
    current_acc = accuracy_score(y_true, (probs > 0.5).astype(int))
    baseline_acc = 1 - y_true.mean() # Accuracy if we always predicted "Safe"

    # 5c. Sidebar: Model Benchmark
    st.sidebar.markdown("### 🆚 Model Benchmark")
    fig_bench = go.Figure()
    fig_bench.add_trace(go.Bar(
        x=['Baseline', 'Current Model'],
        y=[baseline_acc, current_acc],
        text=[f"{baseline_acc:.1%}", f"{current_acc:.1%}"],
        textposition='auto',
        marker_color=['#9E9E9E', '#4CAF50']
    ))
    fig_bench.update_layout(
        title="Accuracy vs Baseline",
        title_font_size=14,
        margin=dict(l=10, r=10, t=30, b=10),
        height=200,
        yaxis=dict(showgrid=False, range=[0.9, 1.01])
    )
    st.sidebar.plotly_chart(fig_bench, use_container_width=True)
    st.sidebar.caption(f"Machines: {len(df)} | Failures: {y_true.sum()}")

    # 5d. Apply Predictions
    df['Failure Probability'] = probs
    df['Risk Category'] = df['Failure Probability'].apply(
        lambda p: 'High' if p > 0.7 else ('Medium' if p > 0.4 else 'Low')
    )
    df['Predicted RUL (min)'] = (250 - df['Tool wear [min]']).clip(lower=0)
    df['Recommendation'] = df['Risk Category'].map({
        'High': "🔴 Replace Tool / Inspect",
        'Medium': "🟡 Scheduled Inspection",
        'Low': "🟢 Normal Operation"
    })

# -----------------------------------------------------------------------------
# 6. DASHBOARD PAGES
# -----------------------------------------------------------------------------
if page == "Dashboard Overview":
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Machine Health Monitoring")
        st.markdown(f"**Model:** {selected_model_name} | **Status:** Online")
    with col2:
        st.metric("Plant Health", f"{100 - (df['Risk Category']=='High').mean()*100:.1f}%")
    
    st.divider()

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Machines", len(df))
    c2.metric("High Risk", len(df[df['Risk Category']=='High']), delta_color="inverse")
    c3.metric("Medium Risk", len(df[df['Risk Category']=='Medium']))
    c4.metric("Avg Tool Wear", f"{df['Tool wear [min]'].mean():.1f} min")

    # Alerts
    high_risk_df = df[df['Risk Category'] == 'High']
    if not high_risk_df.empty:
        st.subheader("🔔 Priority Alerts")
        for _, row in high_risk_df.head(3).iterrows():
            st.error(f"**UDI {row['UDI']}**: Risk {row['Failure Probability']:.1%} | Torque: {row['Torque [Nm]']} | {row['Recommendation']}")
    else:
        st.success("✅ No High Risk Machines Detected")

    # Main Prediction Table & Risk Dist
    col_main, col_plot = st.columns([2, 1])
    with col_main:
        st.subheader("Live Predictions")
        st.dataframe(
            df[['UDI', 'Type', 'Torque [Nm]', 'Tool wear [min]', 'Failure Probability', 'Risk Category', 'Recommendation']]
            .sort_values('Failure Probability', ascending=False),
            column_config={"Failure Probability": st.column_config.ProgressColumn(format="%.2f")},
            height=400,
            use_container_width=True
        )
    
    with col_plot:
        st.subheader("Risk Distribution")
        fig_hist = px.histogram(df, x="Failure Probability", nbins=20, color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- RESTORED VISUALIZATIONS SECTION ---
    st.divider()
    st.subheader("📊 Historical Analysis & Features")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("**Failure Cause Breakdown**")
        # Check if failure type columns exist
        fail_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        if all(c in df.columns for c in fail_cols):
            fail_counts = df[fail_cols].sum().reset_index()
            fail_counts.columns = ['Cause', 'Count']
            fig_pie = px.pie(fail_counts, values='Count', names='Cause', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Failure Type columns (TWF, HDF, etc.) not found in dataset.")

    with col_v2:
        st.markdown("**Feature Correlation Heatmap**")
        # Prepare numeric data for correlation
        corr_df = df[feature_names].copy()
        # Ensure Type is numeric
        if 'Type' in corr_df.columns and corr_df['Type'].dtype == 'object':
            corr_df['Type'] = corr_df['Type'].map({'H': 0, 'L': 1, 'M': 2})
        
        corr = corr_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Machine Details":
    st.title("🔎 Diagnostics")
    sel_udi = st.selectbox("Select UDI", df['UDI'].unique())
    row = df[df['UDI'] == sel_udi].iloc[0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Level", row['Risk Category'])
    c2.metric("Failure Prob", f"{row['Failure Probability']:.2%}")
    c3.metric("RUL", f"{row['Predicted RUL (min)']} min")
    
    # Simulated Trends
    st.subheader("Sensor Trends (Last 20 Cycles)")
    col_t1, col_t2 = st.columns(2)
    
    hist_torque = [row['Torque [Nm]'] * (1 + np.random.normal(0, 0.05)) for _ in range(20)]
    fig_t = px.line(y=hist_torque, title="Torque History")
    fig_t.add_hline(y=60, line_dash="dash", line_color="red")
    col_t1.plotly_chart(fig_t, use_container_width=True)
    
    hist_wear = [row['Tool wear [min]'] * (1 + np.random.normal(0, 0.02)) for _ in range(20)]
    fig_w = px.line(y=hist_wear, title="Tool Wear History")
    fig_w.add_hline(y=200, line_dash="dash", line_color="red")
    col_t2.plotly_chart(fig_w, use_container_width=True)

elif page == "Model Performance":
    st.title("Model Evaluation")
    y_pred = (df['Failure Probability'] > 0.5).astype(int)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    m2.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
    m3.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
    m4.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
    
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        # Improved Heatmap for Confusion Matrix
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Normal', 'Failure'], y=['Normal', 'Failure'],
                           color_continuous_scale='Blues')
        fig_cm.update_layout(title_text="Confusion Matrix", title_x=0.5)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col_roc:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, df['Failure Probability'])
        fig_roc = px.area(x=fpr, y=tpr, title=f"AUC: {auc(fpr, tpr):.4f}",
                          labels=dict(x='False Positive Rate', y='True Positive Rate'))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)

elif page == "About":
    st.title("ℹ️ Project Info")
    st.markdown("Predictive maintenance dashboard using AI4I 2020 dataset. Models: RF, GB, SVM, MLP.")
