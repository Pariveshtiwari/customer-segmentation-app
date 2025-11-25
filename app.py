import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation - Universal", layout="wide")

st.title("🧩 Universal Customer Segmentation App (Any Dataset)")
st.write("Upload any CSV file and select 2 numeric columns for K-Means clustering.")

# ------------------------------------------------------
# Sidebar
# ------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])
show_elbow = st.sidebar.checkbox("📉 Show Elbow Method", value=True)
clusters = st.sidebar.slider("🎯 Number of Clusters (K)", 2, 10, 5)

# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("❌ Your dataset must contain at least TWO numeric columns.")
        st.stop()

    st.sidebar.subheader("🔢 Select Features for Clustering")

    col1 = st.sidebar.selectbox("Select X-axis Feature", numeric_cols)
    col2 = st.sidebar.selectbox("Select Y-axis Feature", numeric_cols)

    # Feature matrix
    X = df[[col1, col2]].values

    # ------------------------------------------------------
    # Elbow Curve
    # ------------------------------------------------------
    if show_elbow:
        st.subheader("📉 Elbow Method")

        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(range(1, 11), wcss, marker="o")
        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("WCSS")
        st.pyplot(fig)

    # ------------------------------------------------------
    # KMeans Clustering
    # ------------------------------------------------------
    st.subheader("🎯 K-Means Clustering Result")

    kmeans = KMeans(n_clusters=clusters, init="k-means++", random_state=42)
    y = kmeans.fit_predict(X)

    # Add cluster column to dataset
    df["Cluster"] = y

    st.dataframe(df.head())

    # ------------------------------------------------------
    # Visualization
    # ------------------------------------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i in range(clusters):
        plt.scatter(
            X[y == i, 0],
            X[y == i, 1],
            s=100,
            color=colors[i],
            label=f"Cluster {i+1}"
        )

    # Plot centroids
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=300,
        c="yellow",
        label="Centroids"
    )

    plt.title("Cluster Visualization")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()
    st.pyplot(fig)

    # ------------------------------------------------------
    # Download Segmented Data
    # ------------------------------------------------------
    st.subheader("⬇ Download Segmented Dataset")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="segmented_dataset.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a dataset to begin.")
