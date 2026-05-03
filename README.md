# 🚀 Customer Segmentation Project

A machine learning project that performs **customer segmentation using K-Means clustering**, with a modular pipeline and interactive visualization (Streamlit-based).

---

## 📌 Overview

This project groups customers into distinct clusters based on features like **Age, Income, and Spending Behavior**.

The goal is to simulate a **real-world business use case**:
- Identify high-value customers
- Understand spending patterns
- Enable targeted marketing strategies

---

## 🎯 Key Features

- 📥 Load dataset from CSV
- 🎯 Feature extraction pipeline
- 🤖 K-Means clustering implementation
- 📉 Elbow Method (WCSS) for optimal cluster selection
- 📊 Cluster visualization
- 💾 Model saving & loading (Joblib)
- 🖥️ Streamlit UI for interaction

---

## 🧠 Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn (KMeans)**
- **Matplotlib**
- **Streamlit**
- **Joblib**

---

## 📂 Project Structure

```bash
Customer-Segmentation-Project/
│
├── data/
│   └── customers.csv
│
├── src/
│   ├── clustering.py        # KMeans + pipeline logic
│   ├── utils.py             # Visualization functions
│
├── app.py                   # Streamlit application
├── requirements.txt
├── README.md
└── kmeans.pkl               # Saved model (generated)
