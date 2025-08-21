from __future__ import annotations
from pathlib import Path
import io
from typing import Dict, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # 设置matplotlib后端为Agg（无头模式）
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
META_DIR = PROJECT_ROOT.parent / "data"  # Keep access to variable description files


@st.cache_resource(show_spinner=False)
def load_model() -> object:
    model_path = MODEL_DIR / "ann_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_training_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load scaled training/test frames for feature order and SHAP background.

    Read with index_col=0 so the first column (saved index) is not treated as a feature.
    """
    x_train_scaled = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
    x_test_scaled = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
    return x_train_scaled, x_test_scaled


@st.cache_data(show_spinner=False)
def load_variable_help() -> Dict[str, str]:
    """Parse the variable description xlsx to extract the text inside parentheses.

    Returns mapping: column_name -> help_text
    """
    meta_path = META_DIR / "Variable .xlsx"
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
            "Number of hospitalizations": "Number of hospitalizations (0=0, 1=At least once)",
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
    """Create a scaler for Age only.

    Priority of sources for mean/std:
    1) data/X_train_notscaled.csv (if present with the expected columns)
    2) Approximate from known ranges using scaled min/max (last resort)
    """
    target_cols = ["Age"]

    # 1) Preferred: explicit unscaled training data
    unscaled_path = DATA_DIR / "X_train_notscaled.csv"
    if unscaled_path.exists():
        df_unscaled = pd.read_csv(unscaled_path, index_col=0)
        if all(col in df_unscaled.columns for col in target_cols):
            scaler = StandardScaler().fit(df_unscaled[target_cols])
            return scaler

    # 2) Last resort: approximate from scaled min/max and known raw ranges
    # This is a coarse approximation if raw means/std are not available.
    if set(target_cols).issubset(x_train_scaled.columns):
        z = x_train_scaled[target_cols].copy()
        approx_mean = np.array([65.0])  # typical default for age
        # sigma derived from span if possible
        try:
            z_span = (z.max() - z.min()).values
            age_sigma = 40.0 / max(z_span[0], 1e-6)  # age range ~45..85 -> span 40
            sigma = np.array([age_sigma])
            scaler = StandardScaler()
            scaler.mean_ = approx_mean
            scaler.scale_ = sigma
            scaler.var_ = sigma ** 2
            scaler.n_features_in_ = 1
            scaler.feature_names_in_ = np.array(target_cols)
            return scaler
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model: object, x_train_scaled: pd.DataFrame):
    """Create and cache a SHAP explainer for the model with optimized background."""
    # 使用k-means采样减少背景数据量，提高速度
    background_size = min(1000, len(x_train_scaled))  # 限制背景样本数量
    background = shap.kmeans(x_train_scaled, background_size)
    return shap.KernelExplainer(lambda X: _model.predict_proba(X)[:, 1], background)



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
                    "Medium physical activity (600-3000 MET-min/week) (2)": 2,
                    "High physical activity (>3000 MET-min/week) (3)": 3,
                },
                "Low physical activity (<600 MET-min/week) (1)",
            )
            hosp_raw = sb(0, "Number of hospitalizations", {"0 (0)": 0, "At least once (1)": 1}, "0 (0)")
            online = sb(1, "Online participation", {"No (0)": 0, "Uses internet (1)": 1}, "No (0)")

            # Add unified comments above the prediction button
            st.markdown("---")
            st.markdown("**Variable Definitions:**", help="Click to view detailed explanations")
            with st.expander("📋 Variable Definitions"):
                st.markdown("""
                **Physical activity level**:MET minutes/week = MET value * days * duration. The sum of MET minutes/week corresponding to the three MET values needs to be calculated.
                                            The MET values are further categorized into three groups: (MET=8.0, such as climbing, running, and farming), (MET=4.0, such as brisk walking and Tai Chi), and (MET=3.3, such as casual walking).
                
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
            "Number of hospitalizations": hosp_raw,  # 现在是分类变量，不需要float转换
            "Online participation": online,
        }

        # Create dataframe and scale numeric fields to match the model's expectations
        sample_df_unscaled = pd.DataFrame([user_raw])
        # Align columns in the correct order
        sample_df_unscaled = sample_df_unscaled[[c for c in feature_names]]

        # Transform numerics using scaler learned from training (unscaled -> scaled)
        numerics = ["Age"]  # Only Age needs scaling now
        scaled_values = sample_df_unscaled.copy()
        if numerics and scaler is not None:
            scaled_values[numerics] = scaler.transform(sample_df_unscaled[numerics])
        else:
            st.warning(
                "Automatic standardization for Age is not available. "
                "Please enter standardized values (z-scores) directly."
            )

        if submitted:
            # 使用最准确有效的预测概率方法
            try:
                # 直接使用模型的predict_proba方法
                proba = float(model.predict_proba(scaled_values)[0, 1])
                predicted_class = int(proba >= 0.5)
            except Exception as e:
                st.error(f"预测失败: {e}")
                return

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
                try:
                    # 使用缓存的SHAP explainer
                    explainer = get_shap_explainer(model, x_train_scaled)
                    shap_values = explainer(scaled_values)
                    
                    # Waterfall plot centered and taking half width
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # 确保matplotlib使用Agg后端（无头模式）
                        plt.switch_backend('Agg')
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        fig = plt.gcf()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)  # 清理图形避免内存泄漏
                    
                    # Add explanation
                    st.info("💡 SHAP plot shows the contribution of each feature to the prediction result. Positive values indicate increased risk, negative values indicate decreased risk.")
                except Exception as e:
                    st.error(f"SHAP分析失败: {e}")
                    st.info("请检查模型和数据是否正确加载。")

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
            - Age is automatically standardized
            - All categorical variables are encoded as numerical values
            - Model is based on training data, actual application requires clinical judgment
            """)

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check if data files and model files exist and are in the correct format.")


if __name__ == "__main__":
    main()


