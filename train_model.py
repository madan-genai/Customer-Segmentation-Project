from clustering import load_data, extract_features, fit_kmeans, save_model

# Load dataset
df = load_data("data/train.csv")

# Choose features (IMPORTANT: same as app)
features = ["annual_income", "spending_score"]

X = extract_features(df, features)

# Train model
model, labels, centers = fit_kmeans(X, n_clusters=5)

# Save model
save_model(model)

print("✅ Model trained and saved to model/kmeans.pkl")