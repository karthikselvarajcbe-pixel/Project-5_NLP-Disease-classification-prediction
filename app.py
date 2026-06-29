import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Clinical Trial Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CACHE DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        r"D:/Project#5/notebooks/clinial_trial_disease_prediction_preprocessed_dataset.csv"
    )

# =====================================================
# CACHE MODELS
# =====================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        "clinical_trial_lr_model.pkl"
    )

    tfidf = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    encoder = joblib.load(
        "label_encoder.pkl"
    )

    return model, tfidf, encoder

# =====================================================
# LOAD
# =====================================================

df = load_data()

model, tfidf, label_encoder = load_models()

# =====================================================
# NLP SETUP
# =====================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# =====================================================
# PREPROCESS FUNCTION
# =====================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color:#F8FAFC;
}

.stButton > button {
    width:100%;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("🩺 Clinical Trial Disease Classification System")

st.markdown("""
Predict disease conditions from clinical trial descriptions
using a Logistic Regression NLP model.
""")

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(
    [
        "🩺 Disease Prediction",
        "📊 Analytics Dashboard"
    ]
)

# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.subheader("Clinical Trial Disease Prediction")

    trial_text = st.text_area(
        "Clinical Trial Description",
        height=250
    )

    if st.button("🔍 Predict Disease"):

        if trial_text.strip():

            cleaned_text = preprocess_text(
                trial_text
            )

            vectorized_text = tfidf.transform(
                [cleaned_text]
            )

            prediction = model.predict(
                vectorized_text
            )

            confidence = (
                model.predict_proba(
                    vectorized_text
                ).max() * 100
            )

            disease = (
                label_encoder
                .inverse_transform(prediction)[0]
            )

            st.success(
                f"Predicted Disease : {disease}"
            )

            # Gauge Chart

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={
                        'text': "Prediction Confidence (%)"
                    },
                    gauge={
                        'axis': {
                            'range': [0, 100]
                        }
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning(
                "Please enter clinical trial text."
            )

# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.subheader(
        "📊 Clinical Trial Analytics Dashboard"
    )

    # ============================================
    # KPI SECTION
    # ============================================

    top3 = (
        df["source_condition_query"]
        .value_counts()
        .head(3)
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Trials",
        f"{len(df):,}"
    )

    c2.metric(
        "Disease Classes",
        df["source_condition_query"].nunique()
    )

    c3.metric(
        "Top Disease",
        top3.index[0]
    )

    c4.metric(
        "Model Accuracy",
        "97.12%"
    )

    st.divider()

    visualization_option = st.selectbox(

        "Select Visualization",

        [

            "Top Disease Categories",

            "Top 3 Diseases",

            "Study Type Distribution",

            "Healthy Volunteers",

            "Summary Length Distribution",

            "Top Medical Terms",

            "Disease vs Study Type Heatmap"

        ]
    )

    # =====================================================
    # TOP DISEASES
    # =====================================================

    if visualization_option == "Top Disease Categories":

        disease_df = (

            df["source_condition_query"]

            .value_counts()

            .head(15)

            .reset_index()

        )

        disease_df.columns = [
            "Disease",
            "Count"
        ]

        fig = px.bar(

            disease_df,

            x="Count",

            y="Disease",

            orientation="h",

            title="Top 15 Disease Categories"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # TOP 3 DISEASES
    # =====================================================

    elif visualization_option == "Top 3 Diseases":

        top3_df = (
            df["source_condition_query"]
            .value_counts()
            .head(3)
            .reset_index()
        )

        top3_df.columns = [
            "Disease",
            "Count"
        ]

        fig = px.pie(
            top3_df,
            names="Disease",
            values="Count",
            hole=0.4,
            title="Top 3 Disease Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # STUDY TYPE
    # =====================================================

    elif visualization_option == "Study Type Distribution":

        study_df = (
            df["study_type"]
            .value_counts()
            .reset_index()
        )

        study_df.columns = [
            "Study Type",
            "Count"
        ]

        fig = px.pie(

            study_df,

            names="Study Type",

            values="Count",

            title="Study Type Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # VOLUNTEERS
    # =====================================================

    elif visualization_option == "Healthy Volunteers":

        volunteer_df = (
            df["healthy_volunteers"]
            .value_counts()
            .reset_index()
        )

        volunteer_df.columns = [
            "Volunteer Status",
            "Count"
        ]

        fig = px.bar(

            volunteer_df,

            x="Volunteer Status",

            y="Count",

            title="Healthy Volunteer Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # SUMMARY LENGTH
    # =====================================================

    elif visualization_option == "Summary Length Distribution":

        temp = df.copy()

        temp["summary_length"] = (

            temp["brief_summaries"]

            .fillna("")

            .str.split()

            .str.len()

        )

        fig = px.histogram(

            temp,

            x="summary_length",

            nbins=50,

            title="Clinical Trial Summary Length"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # TOP TERMS
    # =====================================================

    elif visualization_option == "Top Medical Terms":

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=20
        )

        X = vectorizer.fit_transform(
            df["brief_summaries"]
            .fillna("")
        )

        terms_df = pd.DataFrame({

            "Term":
            vectorizer.get_feature_names_out(),

            "Count":
            X.sum(axis=0).A1

        })

        terms_df = terms_df.sort_values(
            "Count",
            ascending=False
        )

        fig = px.bar(

            terms_df,

            x="Term",

            y="Count",

            title="Top Medical Terms"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # HEATMAP
    # =====================================================

    elif visualization_option == "Disease vs Study Type Heatmap":

        import seaborn as sns
        import matplotlib.pyplot as plt

        top10 = (
            df["source_condition_query"]
            .value_counts()
            .head(10)
            .index
        )

        heat_df = df[
            df["source_condition_query"]
            .isin(top10)
        ]

        pivot = pd.crosstab(

            heat_df["source_condition_query"],

            heat_df["study_type"]

        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        sns.heatmap(

            pivot,

            annot=True,

            fmt="d",

            cmap="Blues",

            ax=ax

        )

        st.pyplot(fig)