from __future__ import annotations

import math
from pathlib import Path
import io
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------------------
# Paths and cached loaders
# --------------------------------------------------------------------------------------

# Use data from streamlit directory
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
META_DIR = PROJECT_ROOT.parent / "CHARLS_18_pain"  # Keep access to variable description files


@st.cache_resource(show_spinner=False)
def load_model() -> object:
    model_path = MODEL_DIR / "svm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_training_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load scaled training/test frames for feature order and SHAP background."""
    x_train_scaled = pd.read_csv(DATA_DIR / "X_train.csv")
    x_test_scaled = pd.read_csv(DATA_DIR / "X_test.csv")
    # Ensure index is not treated as a feature if present
    if x_train_scaled.columns[0] in ("Unnamed: 0", ""):
        x_train_scaled = x_train_scaled.drop(columns=[x_train_scaled.columns[0]])
    if x_test_scaled.columns[0] in ("Unnamed: 0", ""):
        x_test_scaled = x_test_scaled.drop(columns=[x_test_scaled.columns[0]])
    return x_train_scaled, x_test_scaled


@st.cache_data(show_spinner=False)
def load_variable_help() -> Dict[str, str]:
    """Parse the variable description csv to extract the text inside parentheses.

    Returns mapping: column_name -> help_text
    """
    meta_path = META_DIR / "Variable .csv"
    if not meta_path.exists():
        # If variable description file is not found, provide default descriptions
        default_help = {
            "Sex": "Gender (0=Female, 1=Male)",
            "Age": "Age (45-85 years)",
            "Education level": "Education level (1=Below high school, 2=High school and above)",
            "Drinking status": "Drinking status (0=No, 1=Yes)",
            "Diabetes": "Diabetes (0=No, 1=Yes)",
            "Stroke": "Stroke (0=No, 1=Yes)",
            "Kidney disease": "Kidney disease (0=No, 1=Yes)",
            "Emotional problems": "Emotional problems (0=No, 1=Yes)",
            "Memory-related disease": "Memory-related disease (0=No, 1=Yes)",
            "Asthma": "Asthma (0=No, 1=Yes)",
            "Fall down": "Fall down (0=No, 1=Yes)",
            "Depression": "Depression (0=No, 1=Yes)",
            "Life satisfaction": "Life satisfaction (1=Low, 2=Medium, 3=High)",
            "Self perceived health status": "Self perceived health status (1=Poor, 2=Fair, 3=Good)",
            "Physical activity level": "Physical activity level (1=Low, 2=Medium, 3=High)",
            "Number of hospitalizations": "Number of hospitalizations (≥0)",
            "Online participation": "Online participation (0=No, 1=Uses internet)"
        }
        return default_help
    
    rows = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"):
        try:
            rows = pd.read_csv(meta_path, header=None, names=["name", "desc"], encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if rows is None:
        # Final fallback: ignore errors while decoding to avoid crash
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        rows = pd.read_csv(io.StringIO(text), header=None, names=["name", "desc"]) 
    # Normalize names to match X columns
    rows["name"] = rows["name"].astype(str).str.strip()
    rows["desc"] = rows["desc"].astype(str).str.strip().str.strip('"')
    return dict(zip(rows["name"], rows["desc"]))


def build_numeric_scaler(x_train_scaled: pd.DataFrame) -> Optional[StandardScaler]:
    """Create a scaler for Age and Number of hospitalizations.

    Priority of sources for mean/std:
    1) data/X_train_notscaled.csv (if present with the expected columns)
    2) data/train_data_notscaled.csv (fallback; map column names case-insensitively)
    3) Approximate from known ranges using scaled min/max (last resort)
    """
    target_cols = ["Age", "Number of hospitalizations"]

    # 1) Preferred: explicit unscaled training data
    unscaled_path = DATA_DIR / "X_train_notscaled.csv"
    if unscaled_path.exists():
        df_unscaled = pd.read_csv(unscaled_path)
        if df_unscaled.columns[0] in ("Unnamed: 0", ""):
            df_unscaled = df_unscaled.drop(columns=[df_unscaled.columns[0]])
        if all(col in df_unscaled.columns for col in target_cols):
            scaler = StandardScaler().fit(df_unscaled[target_cols])
            return scaler

    # 2) Fallback to broader raw training file if available
    generic_path = DATA_DIR / "train_data_notscaled.csv"
    if generic_path.exists():
        raw_df = pd.read_csv(generic_path)
        # Build name mapping (case-insensitive, strip spaces/underscores)
        normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in raw_df.columns}
        age_col = normalized.get("age")
        hosp_col = normalized.get("numberofhospitalizations") or normalized.get("number_of_hospitalizations")
        selected = []
        if age_col:
            selected.append(age_col)
        if hosp_col:
            selected.append(hosp_col)
        if len(selected) == 2:
            scaler = StandardScaler().fit(raw_df[[age_col, hosp_col]].rename(columns={age_col: "Age", hosp_col: "Number of hospitalizations"}))
            return scaler

    # 3) Last resort: approximate from scaled min/max and known raw ranges
    # This is a coarse approximation if raw means/std are not available.
    z = x_train_scaled[target_cols].copy()
    if set(target_cols).issubset(z.columns):
        approx_mean = np.array([65.0, 0.0])  # typical defaults
        # sigma derived from span if possible
        try:
            z_span = (z.max() - z.min()).values
            age_sigma = 40.0 / max(z_span[0], 1e-6)  # age range ~45..85 -> span 40
            hosp_sigma = 5.0 / max(z_span[1], 1e-6)   # assume 0..5 typical span
            sigma = np.array([age_sigma, hosp_sigma])
            scaler = StandardScaler()
            scaler.mean_ = approx_mean
            scaler.scale_ = sigma
            scaler.var_ = sigma ** 2
            scaler.n_features_in_ = 2
            scaler.feature_names_in_ = np.array(target_cols)
            return scaler
        except Exception:
            return None
    return None


def probability_predict_fn(model: object):
    """Return a function f(X)->prob_class1 robust to models without predict_proba."""
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return lambda X: 1.0 / (1.0 + np.exp(-model.decision_function(X)))
    # Fallback to predict; coerce to float
    return lambda X: model.predict(X).astype(float)


def render_title():
    st.set_page_config(
        page_title="Model for Predicting ADL Disability Risk in Patients with Low Back Pain",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Model for Predicting ADL Disability Risk in Patients with Low Back Pain")
    st.caption(
        "Enter patient characteristics to estimate risk and explain predictions through SHAP waterfall plots."
    )


def main():
    render_title()

    try:
        # Load assets
        model = load_model()
        x_train_scaled, x_test_scaled = load_training_frames()
        variable_help = load_variable_help()
        scaler = build_numeric_scaler(x_train_scaled)

        feature_names: List[str] = list(x_train_scaled.columns)
        if len(feature_names) != 17:
            st.warning(
                f"Detected {len(feature_names)} features in training data, but expected 17. "
                "Please ensure X_train.csv contains the correct 17 columns."
            )

        # Sidebar / Form for inputs
        with st.form("input_form"):
            st.subheader("Patient Information Input")
            cols = st.columns(3)

            # Utility for selectboxes
            def sb(idx: int, label: str, options: Dict[str, int], default_key: str) -> int:
                help_text = variable_help.get(label, "")
                choice = cols[idx % 3].selectbox(
                    f"{label}",
                    options=list(options.keys()),
                    index=list(options.keys()).index(default_key),
                    help=help_text,
                )
                return options[choice]

            # Inputs
            sex = sb(0, "Sex", {"Female (0)": 0, "Male (1)": 1}, "Female (0)")
            age_raw = cols[1].number_input(
                "Age",
                min_value=45,
                max_value=85,
                value=65,
                step=1,
                help=variable_help.get("Age", "(45~85 years)"),
            )
            edu = sb(2, "Education level", {"Below high school (1)": 1, "High school and above (2)": 2}, "Below high school (1)")
            drinking = sb(0, "Drinking status", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            db = sb(1, "Diabetes", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            stroke = sb(2, "Stroke", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            kidney = sb(0, "Kidney disease", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            emo = sb(1, "Emotional problems", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            mem = sb(2, "Memory-related disease", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            asthma = sb(0, "Asthma", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            fall = sb(1, "Fall down", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            dep = sb(2, "Depression", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
            life_sat = sb(
                0,
                "Life satisfaction",
                {"Low (1)": 1, "Medium (2)": 2, "High (3)": 3},
                "Medium (2)",
            )
            selfhea = sb(
                1,
                "Self perceived health status",
                {"Poor (1)": 1, "Fair (2)": 2, "Good (3)": 3},
                "Fair (2)",
            )
            pa = sb(
                2,
                "Physical activity level",
                {
                    "Low physical activity (<600 MET-min/week) (1)": 1,
                    "Medium physical activity (600-3000) (2)": 2,
                    "High physical activity (>3000) (3)": 3,
                },
                "Low physical activity (<600 MET-min/week) (1)",
            )
            hosp_raw = cols[0].number_input(
                "Number of hospitalizations",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                help=variable_help.get("Number of hospitalizations", "(≥0)"),
            )
            online = sb(1, "Online participation", {"No (0)": 0, "Uses internet (1)": 1}, "No (0)")

            # Add unified comments above the prediction button
            st.markdown("---")
            st.markdown("**Variable Definitions:**", help="Click to view detailed explanations")
            with st.expander("📋 Variable Definitions"):
                st.markdown("""
                **Physical activity level**: The MET values are further categorized into three groups: (MET=8.0, such as climbing, running, and farming), (MET=4.0, such as brisk walking and Tai Chi), and (MET=3.3, such as casual walking).
                MET minutes/week = MET value * days * duration. The sum of MET minutes/week corresponding to the three MET values needs to be calculated.
                
                **Number of hospitalizations**: How many times have you received inpatient care during the past year?
                """)
            
            submitted = st.form_submit_button("Start Prediction", use_container_width=True)

        # Prepare the single-row input in the exact feature order
        user_raw: Dict[str, float] = {
            "Sex": sex,
            "Age": float(age_raw),
            "Education level": edu,
            "Drinking status": drinking,
            "Diabetes": db,
            "Stroke": stroke,
            "Kidney disease": kidney,
            "Emotional problems": emo,
            "Memory-related disease": mem,
            "Asthma": asthma,
            "Fall down": fall,
            "Depression": dep,
            "Life satisfaction": life_sat,
            "Self perceived health status": selfhea,
            "Physical activity level": pa,
            "Number of hospitalizations": float(hosp_raw),
            "Online participation": online,
        }

        # Create dataframe and scale numeric fields to match the model's expectations
        sample_df_unscaled = pd.DataFrame([user_raw])
        # Align columns in the correct order
        sample_df_unscaled = sample_df_unscaled[[c for c in feature_names]]

        # Transform numerics using scaler learned from training (unscaled -> scaled)
        numerics = [c for c in ["Age", "Number of hospitalizations"] if c in sample_df_unscaled.columns]
        scaled_values = sample_df_unscaled.copy()
        if numerics and scaler is not None:
            scaled_values[numerics] = scaler.transform(sample_df_unscaled[numerics])
        else:
            st.warning(
                "Automatic standardization for Age and Number of hospitalizations is not available. "
                "Please enter standardized values (z-scores) directly."
            )

        if submitted:
            pred_fn = probability_predict_fn(model)
            proba = float(pred_fn(scaled_values)[0])
            predicted_class = int(proba >= 0.5)

            # Display prediction results
            st.subheader("Prediction Results")
            
            # Use more intuitive display method
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", "At Risk" if predicted_class == 1 else "Low Risk")
                st.metric("Risk Probability", f"{proba:.1%}")
            
            with col2:
                if predicted_class == 1:
                    st.error("⚠️ Prediction: ADL Disability Risk Present")
                else:
                    st.success("✅ Prediction: Low ADL Disability Risk")
                
                st.info(f"Detailed Probability: At Risk {proba:.1%} | Low Risk {(1.0-proba):.1%}")

            # Explain with SHAP Waterfall
            st.subheader("SHAP Feature Importance Explanation")
            with st.spinner("Calculating SHAP explanation..."):
                # Background: small, for performance
                background = x_train_scaled.sample(
                    n=min(100, len(x_train_scaled)), random_state=6
                )
                explainer = shap.Explainer(pred_fn, background, feature_names=feature_names)
                explanation = explainer(scaled_values)

                # Waterfall plot centered and taking half width
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    shap.plots.waterfall(explanation[0], max_display=10, show=False)
                    fig = plt.gcf()
                    st.pyplot(fig, use_container_width=True)
                
                # Add explanation
                st.info("💡 SHAP plot shows the contribution of each feature to the prediction result. Positive values indicate increased risk, negative values indicate decreased risk.")

            st.warning("⚠️ Important Reminder: This tool provides data-driven estimates and should not replace professional medical advice.")

        # Add usage instructions
        with st.expander("📖 Usage Instructions"):
            st.markdown("""
            **Model Description:**
            - This model uses machine learning algorithms to predict ADL (Activities of Daily Living) disability risk in patients with low back pain
            - Input 17 clinically relevant variables including demographics, medical history, lifestyle, etc.
            
            **Result Interpretation:**
            - Predicted Class: 0=Low Risk, 1=At Risk
            - Risk Probability: Value between 0-1, closer to 1 indicates higher risk
            - SHAP Plot: Shows the impact of each feature on the prediction result
            
            **Notes:**
            - Age and number of hospitalizations are automatically standardized
            - All categorical variables are encoded as numerical values
            - Model is based on training data, actual application requires clinical judgment
            """)

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check if data files and model files exist and are in the correct format.")


if __name__ == "__main__":
    main()


