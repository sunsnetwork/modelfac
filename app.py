import streamlit as st
import pandas as pd
import tempfile
import xgboost as xgb
from model import ModelFac
from streamlit_authenticator import Authenticate

# --- User Auth Setup ---
config = {
    'credentials': {
        'usernames': {
            'demo_user': {
                'email': 'demo@example.com',
                'name': 'Demo User',
                'password': 'demo'  # Use hashed passwords in production!
            },
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'modelfac_auth',
        'name': 'modelfac_cookie'
    },
    'preauthorized': {
        'emails': ['demo@example.com']
    }
}

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, auth_status, username = authenticator.login('Login', 'main')

if auth_status:
    st.set_page_config(page_title="ModelFac: Propensity Modeling", layout="wide")
    st.title("📊 ModelFac: Propensity Modeling Agent")
    st.markdown("""
    Upload your **training** and **testing** datasets (CSV format). ModelFac will train a propensity model using XGBoost, 
    generate performance insights (SHAP, lift chart), and return a scored dataset.
    """)

    authenticator.logout('Logout', 'sidebar')
    st.sidebar.success(f"Welcome {name}!")

    with st.sidebar:
        st.header("Upload Files")
        train_file = st.file_uploader("Training Data (CSV)", type="csv", key="train")
        test_file = st.file_uploader("Testing Data (CSV)", type="csv", key="test")

    model = ModelFac()

    if train_file and test_file:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        st.subheader("🔍 Training Data Preview")
        st.dataframe(df_train.head(), use_container_width=True)

        target_col = st.selectbox("🎯 Select Target Variable", df_train.columns)

        exclude_cols = st.multiselect(
            "🚫 Select Columns to Exclude from Inputs", 
            options=[col for col in df_train.columns if col != target_col]
        )

        if st.button("🚀 Train Model & Score Test Data"):
            with st.spinner("Training model and generating report..."):
                df_train_model = df_train.drop(columns=exclude_cols)
                df_test_model = df_test.drop(columns=exclude_cols)

                auc = model.train(df_train_model, target_col)
                scored_df = model.predict(df_test_model)

                with tempfile.TemporaryDirectory() as tmpdir:
                    shap_path = model.explain(df_train_model, tmpdir)
                    lift_path = model.lift_chart(df_train_model.assign(score=model.model.predict(xgb.DMatrix(df_train_model[model.features]))), tmpdir)

                    st.success(f"Model trained successfully! AUC = {auc:.4f}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(shap_path, caption="SHAP Summary Plot", use_column_width=True)

                    with col2:
                        st.image(lift_path, caption="Lift Chart by Decile", use_column_width=True)

                    st.subheader("📥 Scored Test Data")
                    st.dataframe(scored_df.head(), use_container_width=True)

                    csv_download = scored_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Scored Data", csv_download, file_name="scored_test_data.csv", mime="text/csv")
    else:
        st.info("Please upload both training and testing CSV files to begin.")

elif auth_status is False:
    st.error("Invalid username or password")
elif auth_status is None:
    st.warning("Please enter your username and password")