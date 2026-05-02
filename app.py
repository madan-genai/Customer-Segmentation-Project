import streamlit as st
import matplotlib.pyplot as plt
from clustering import load_data, extract_features, load_model, calculate_wcss
from utils import plot_cluster_counts, visualize_clusters

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation (Pretrained Model)")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if not uploaded_file:
    st.sidebar.info("Upload a dataset to proceed.")
    st.stop()

# Load data
data = load_data(uploaded_file)
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Feature selection
feature_options = list(data.columns)

selected_features = st.sidebar.multiselect(
    "Select exactly 2 features:",
    options=feature_options,
    default=feature_options[1:3]
)

if len(selected_features) != 2:
    st.error("Please select exactly 2 features.")
    st.stop()

# Load pretrained model
model = load_model("kmeans.pkl")

# Run clustering
if st.sidebar.button("Run Clustering"):

    X = extract_features(data, selected_features)

    # Elbow (visual only)
    wcss = calculate_wcss(X)
    fig_elbow, ax = plt.subplots()
    ax.plot(range(1, len(wcss) + 1), wcss, marker='o')
    ax.set_title("Elbow Method")
    st.pyplot(fig_elbow)

    # Predict clusters
    labels = model.predict(X)
    centers = model.cluster_centers_

    data["Cluster"] = labels

    # Plot clusters
    st.subheader("Cluster Visualization")
    fig_clusters = visualize_clusters(X, labels, centers)
    st.pyplot(fig_clusters)

    # Cluster count
    st.subheader("Cluster Distribution")
    fig_counts = plot_cluster_counts(labels)
    st.pyplot(fig_counts)

    st.success("✅ Clustering completed using pretrained model!")