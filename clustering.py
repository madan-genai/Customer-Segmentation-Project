import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List
import joblib


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def extract_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return df[feature_cols].to_numpy()


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, np.ndarray]:
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    return kmeans, labels, kmeans.cluster_centers_


def calculate_wcss(X: np.ndarray, max_clusters: int = 10) -> List[float]:
    wcss = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss


# 🔥 NEW: Save & Load model
def save_model(model, path: str = "kmeans.pkl") -> None:
    joblib.dump(model, path)


def load_model(path: str = "kmeans.pkl"):
    return joblib.load(path)