import joblib

model = joblib.load("models/fraud_model.pkl")

print("Model loaded successfully!")
print("Model type:", type(model))
print("Expected features:", model.n_features_in_)