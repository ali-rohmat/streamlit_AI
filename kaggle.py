#!/usr/bin/env python3
"""
Streamlit App untuk Klasifikasi Jamur dengan KNN (port dari kaggle_knn.ipynb).
Dataset: mushrooms.csv (8124 samples, 22 categorical features)
Model: KNeighborsClassifier dengan tuning K terbaik.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load dan tampilkan dataset mushrooms.csv."""
    url = "https://raw.githubusercontent.com/ali-rohmat/streamlit_AI/refs/heads/main/mushrooms.csv"
    df = pd.read_csv(url)
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Label encode semua kolom kategorik."""
    le = LabelEncoder()
    df_encoded = df.copy()
    for col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']
    return X, y, le


@st.cache_data
def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[KNeighborsClassifier, float, int, pd.DataFrame, pd.Series, pd.Series]:
    """Train KNN dan cari K terbaik menggunakan elbow method."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    k_range = range(1, 21)
    test_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        test_scores.append(accuracy_score(y_test, knn.predict(X_test)))
    
    best_k = k_range[np.argmax(test_scores)]
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    
    return best_model, accuracy, best_k, pd.DataFrame({
        'k': list(k_range),
        'test_accuracy': test_scores
    }), y_test, best_model.predict(X_test)


def plot_confusion_matrix(cm):
    """Plot confusion matrix menggunakan plotly."""
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Edible', 'Poisonous'],
                    y=['Edible', 'Poisonous'],
                    color_continuous_scale='Blues')
    fig.update_layout(title="Confusion Matrix")
    return fig


def main():
    st.set_page_config(
        page_title="KNN Mushroom Classification",
        page_icon="🍄",
        layout="wide"
    )
    
    st.title("🍄 Klasifikasi Jamur: Edible vs Poisonous dengan KNN")
    st.markdown("Prediksi apakah jamur **beracun (p)** atau **bisa dimakan (e)** berdasarkan 22 ciri.")
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Sample", len(df))
        st.metric("Jumlah Features", df.shape[1] - 1)
    with col2:
        class_dist = df['class'].value_counts()
        st.metric("Edible (e)", class_dist.get('e', 0))
        st.metric("Poisonous (p)", class_dist.get('p', 0))
    
    # Preprocessing dan training
    with st.spinner("Preprocessing dan training model..."):
        X, y, le = preprocess_data(df)
        model, accuracy, best_k, elbow_df, y_test, y_pred = train_model(X, y)
    
    # Metrics
    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best K", best_k)
        st.metric("Test Accuracy", f"{accuracy:.2%}")
    
    # Elbow plot
    fig_elbow = px.line(elbow_df, x='k', y='test_accuracy', 
                       title="Elbow Method - Test Accuracy vs K",
                       markers=True)
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)
    
    # Prediction interface
    st.subheader("🔮 Prediksi Jamur Baru")
    st.markdown("Pilih ciri-ciri jamur (0-encode value dari kategori):")
    
    features = X.columns.tolist()
    input_data = {}
    
    for i, feature in enumerate(features):
        min_val, max_val = int(X[feature].min()), int(X[feature].max())
        input_data[feature] = st.slider(
            f"{feature.replace('-', ' ').title()}", 
            min_value=min_val, 
            max_value=max_val, 
            value=min_val,
            step=1,
            format="d",
            key=f"slider_{feature}"
        )

    

    
    if st.button("🍄 Prediksi!", type="primary"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        label = "✅ **Edible (bisa dimakan)**" if prediction == 0 else "⚠️ **Poisonous (beracun)**"
        st.success(label)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("P(Edible)", f"{probability[0]:.1%}")
        with col2:
            st.metric("P(Poisonous)", f"{probability[1]:.1%}")
    
    # Dataset preview
    if st.checkbox("📋 Lihat Dataset (sample 1000)"):
        st.dataframe(df.head(1000))
    
    # Instructions
    with st.expander("🚀 Cara Menjalankan"):
        st.code("""
pip install -r requirements.txt
streamlit run kaggle.py
        """)
    
    st.info("Model KNN sangat akurat (>99%) karena dataset categorical dengan LabelEncoder.")


if __name__ == "__main__":
    main()
