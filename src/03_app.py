"""Streamlit UI for the diabetes-prediction project.

Two tabs:
  - Predict: enter health indicators, pick a model, get a risk prediction with
    a gauge, a per-case explanation of which features drove it, and a
    side-by-side comparison of what all four algorithms predict.
  - Results: the comparison table and figures produced by 02_models.py, for
    live reference during the paper defense.
"""
import base64
import json
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_dedup.csv")
LOGO_PATH = os.path.join(BASE_DIR, "src", "assets", "logo.svg")

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
}

AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+",
}
GENHLTH_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION_LABELS = {
    1: "Never attended / kindergarten only", 2: "Elementary",
    3: "Some high school", 4: "High school graduate",
    5: "Some college / technical school", 6: "College graduate",
}
INCOME_LABELS = {
    1: "< $10k", 2: "$10k-15k", 3: "$15k-20k", 4: "$20k-25k",
    5: "$25k-35k", 6: "$35k-50k", 7: "$50k-75k", 8: "$75k+",
}

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="st-"], [class*="css-"] { font-size: 1.15rem; }
    p, li, label, div[data-testid="stMarkdownContainer"] { font-size: 1.15rem; }
    h1 { font-size: 2.6rem; }
    h2 { font-size: 2.0rem; }
    h3 { font-size: 1.6rem; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; }
    div[data-testid="stMetricLabel"] { font-size: 1.2rem; }
    .stDataFrame { font-size: 1.05rem; }
    button[data-baseweb="tab"] { font-size: 1.4rem !important; }
    button[data-baseweb="tab"] p { font-size: 1.4rem !important; font-weight: 600; }
    table { font-size: 1.2rem !important; }
    table th, table td { padding: 0.6rem 0.9rem !important; }
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] *,
    div[data-testid="stWidgetLabel"] p { font-size: 1.2rem !important; font-weight: 500 !important; }
    div[data-testid="stCaptionContainer"] { font-size: 1.0rem !important; }
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary { font-size: 1.3rem !important; font-weight: 600; }
    div[data-baseweb="select"] * { font-size: 1.15rem !important; }
    ul[data-baseweb="menu"] li, ul[data-baseweb="menu"] li * { font-size: 1.15rem !important; }
    .st-emotion-cache-1puwf6r p { font-size: 20px !important; }
    [data-testid="stHeaderActionElements"] { display: none !important; }
    .field-label { font-size: 20px !important; font-weight: 500; margin-bottom: 0.25rem; color: #111827; }
    [data-testid="stDecoration"],
    .st-emotion-cache-1dp5vir.ezrtsby1 { background-color: #5540af !important; background-image: none !important; }
    [data-testid="stElementToolbar"],
    .st-emotion-cache-fvjkey.e2wxzia2 { display: none !important; }
    div[data-testid="stImage"], div[data-testid="stImage"] img { margin: 0 !important; padding: 0 !important; background: transparent !important; }
    [data-testid="stFullScreenFrame"] { padding: 0 !important; }
    .app-header { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0; }
    .app-header img { width: 60px; height: 60px; display: block; }
    .app-header h1 { margin: 0 !important; padding: 0 !important; color: #5540af !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    models = {name: joblib.load(os.path.join(MODELS_DIR, fname)) for name, fname in MODEL_FILES.items()}
    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        features = json.load(f)
    return models, features


@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_results():
    out = {}
    for name in ["comparison_test_set", "crossval_scores", "feature_importance_rf", "coefficients_logreg", "permutation_importance"]:
        path = os.path.join(RESULTS_DIR, f"{name}.csv")
        if os.path.exists(path):
            out[name] = pd.read_csv(path)
    return out


def risk_gauge(probability):
    color = "#2ca02c" if probability < 0.4 else "#ff9f1c" if probability < 0.7 else "#d62728"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#e8f5e9"},
                    {"range": [40, 70], "color": "#fff3e0"},
                    {"range": [70, 100], "color": "#fdecea"},
                ],
            },
            title={"text": "Diabetes risk"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def local_explanation(model, input_df, reference_df, features, top_n=6):
    """Occlusion-based local explanation: replace each feature with its
    population median/mode one at a time and measure the resulting change
    in predicted probability. Works for any of the four model types."""
    baseline = model.predict_proba(input_df)[0, 1]
    rows = []
    for feat in features:
        baseline_value = reference_df[feat].median()
        modified = input_df.copy()
        modified[feat] = baseline_value
        modified_proba = model.predict_proba(modified)[0, 1]
        delta = baseline - modified_proba
        rows.append({"feature": feat, "delta": delta, "value": input_df[feat].iloc[0]})
    exp_df = pd.DataFrame(rows).sort_values("delta", key=abs, ascending=False).head(top_n)
    return exp_df, baseline


def render_predict_tab(models, features, reference_df):
    st.subheader("Enter health indicators")

    def field_label(text):
        st.markdown(f'<p class="field-label">{text}</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        field_label("BMI")
        bmi = st.slider("BMI", 12, 60, 28, label_visibility="collapsed")
        field_label("Age category")
        age_code = st.selectbox("Age category", list(AGE_LABELS.keys()), format_func=lambda k: AGE_LABELS[k], index=8, label_visibility="collapsed")
        field_label("General health")
        genhlth = st.selectbox("General health", list(GENHLTH_LABELS.keys()), format_func=lambda k: GENHLTH_LABELS[k], index=2, label_visibility="collapsed")
        field_label("Days of poor mental health (past 30)")
        menthlth = st.slider("Days of poor mental health (past 30)", 0, 30, 0, label_visibility="collapsed")
        field_label("Days of poor physical health (past 30)")
        physhlth = st.slider("Days of poor physical health (past 30)", 0, 30, 0, label_visibility="collapsed")
        field_label("Sex")
        sex = st.radio("Sex", [0, 1], format_func=lambda v: "Female" if v == 0 else "Male", horizontal=True, label_visibility="collapsed")

    with col2:
        highbp = st.toggle("High blood pressure", value=False)
        highchol = st.toggle("High cholesterol", value=False)
        cholcheck = st.toggle("Cholesterol check in past 5 years", value=True)
        smoker = st.toggle("Smoked ≥100 cigarettes lifetime", value=False)
        stroke = st.toggle("Ever had a stroke", value=False)
        heartdisease = st.toggle("Coronary heart disease / heart attack", value=False)
        diffwalk = st.toggle("Serious difficulty walking", value=False)

    with col3:
        physactivity = st.toggle("Physical activity in past 30 days", value=True)
        fruits = st.toggle("Eats fruit ≥1x/day", value=True)
        veggies = st.toggle("Eats vegetables ≥1x/day", value=True)
        hvyalcohol = st.toggle("Heavy alcohol consumption", value=False)
        anyhealthcare = st.toggle("Has any healthcare coverage", value=True)
        nodoccost = st.toggle("Could not see doctor due to cost", value=False)
        field_label("Education level")
        education = st.selectbox("Education level", list(EDUCATION_LABELS.keys()), format_func=lambda k: EDUCATION_LABELS[k], index=4, label_visibility="collapsed")
        field_label("Income level")
        income = st.selectbox("Income level", list(INCOME_LABELS.keys()), format_func=lambda k: INCOME_LABELS[k], index=5, label_visibility="collapsed")

    field_label("Model used for the main prediction")
    model_name = st.selectbox("Model used for the main prediction", list(models.keys()), label_visibility="collapsed")

    values = {
        "HighBP": int(highbp), "HighChol": int(highchol), "CholCheck": int(cholcheck),
        "BMI": bmi, "Smoker": int(smoker), "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heartdisease), "PhysActivity": int(physactivity),
        "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(hvyalcohol),
        "AnyHealthcare": int(anyhealthcare), "NoDocbcCost": int(nodoccost), "GenHlth": genhlth,
        "MentHlth": menthlth, "PhysHlth": physhlth, "DiffWalk": int(diffwalk), "Sex": sex,
        "Age": age_code, "Education": education, "Income": income,
    }
    input_df = pd.DataFrame([values])[features]

    if st.button("Predict", type="primary"):
        model = models[model_name]
        proba = model.predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)

        res_col, gauge_col = st.columns([1, 1])
        with res_col:
            st.metric("Prediction", "Diabetes risk" if pred == 1 else "No diabetes risk")
            st.metric("Predicted probability", f"{proba:.1%}")
            st.caption(f"Model: {model_name}")
        with gauge_col:
            st.plotly_chart(risk_gauge(proba), use_container_width=True)

        st.markdown("#### What drove this prediction")
        exp_df, _ = local_explanation(model, input_df, reference_df, features)
        exp_df["direction"] = exp_df["delta"].apply(
            lambda d: "increases risk" if d > 1e-4 else "decreases risk" if d < -1e-4 else "no clear effect"
        )
        display_exp = exp_df[["feature", "value", "direction", "delta"]].rename(
            columns={"delta": "impact on probability"}
        )
        display_exp["impact on probability"] = display_exp["impact on probability"].apply(lambda v: f"{v:+.3f}")
        st.table(display_exp.set_index("feature"))
        st.caption(
            "Impact = change in predicted probability when this feature is replaced by the population median, "
            "holding everything else fixed. Larger magnitude = bigger driver of this specific prediction."
        )

        st.markdown("#### What each model predicts for this input")
        rows = []
        for name, m in models.items():
            p = m.predict_proba(input_df)[0, 1]
            rows.append({"model": name, "predicted probability": p, "prediction": "Diabetes" if p >= 0.5 else "No diabetes"})
        compare_df = pd.DataFrame(rows).sort_values("predicted probability", ascending=False)
        compare_df["predicted probability"] = compare_df["predicted probability"].apply(lambda v: f"{v:.1%}")
        st.table(compare_df.set_index("model"))


def render_results_tab(results):
    st.subheader("Test-set comparison (held-out, single final evaluation)")
    if "comparison_test_set" in results:
        df = results["comparison_test_set"].copy()
        pct_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in df.columns]
        df[pct_cols] = df[pct_cols].round(4)
        st.table(df.set_index("model"))

    st.subheader("5-fold cross-validation on the training set (mean ± std)")
    if "crossval_scores" in results:
        st.table(results["crossval_scores"].set_index("model"))

    st.subheader("ROC curves")
    roc_path = os.path.join(FIG_DIR, "fig07_roc_curves.png")
    if os.path.exists(roc_path):
        roc_col, _ = st.columns([2, 1])
        with roc_col:
            st.image(roc_path, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model comparison")
        p = os.path.join(FIG_DIR, "fig06_model_comparison.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)
    with c2:
        st.subheader("Confusion matrices")
        p = os.path.join(FIG_DIR, "fig08_confusion_matrices.png")
        if os.path.exists(p):
            st.image(p, use_container_width=True)

    st.subheader("Feature importance (Random Forest, Logistic Regression, permutation)")
    p = os.path.join(FIG_DIR, "fig09_feature_importance.png")
    if os.path.exists(p):
        st.image(p, use_container_width=True)

    with st.expander("Exploratory data analysis figures"):
        eda_figs = sorted(f for f in os.listdir(FIG_DIR) if f.startswith("fig0") and int(f[3:5]) <= 5)
        for f in eda_figs:
            st.image(os.path.join(FIG_DIR, f), use_container_width=True)


def main():
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div class="app-header"><img src="data:image/svg+xml;base64,{logo_b64}" alt="logo">'
        f'<h1>Diabetes Risk Prediction</h1></div>',
        unsafe_allow_html=True,
    )
    st.caption("BRFSS 2015 health-indicators dataset · Decision Tree, Random Forest, Logistic Regression, KNN")

    models, features = load_models()
    reference_df = load_reference_data()
    results = load_results()

    tab_predict, tab_results = st.tabs(["Predict", "Results"])
    with tab_predict:
        render_predict_tab(models, features, reference_df)
    with tab_results:
        render_results_tab(results)


if __name__ == "__main__":
    main()
