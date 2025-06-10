import joblib # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import numpy as np # type: ignore
import pickle

try:
    # Attempt to load the model from model.pkl
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded from model.pkl successfully.")
except Exception as e:
    print("❌ Failed to load model.pkl:", e)
    print("🔁 Generating a dummy model instead...")

    # Train dummy model if loading fails
    X = np.random.rand(100, 14)  # 14 input features
    y = np.random.randint(0, 17, 100)  # 17 possible career classes
    model = RandomForestClassifier()
    model.fit(X, y)

# Save the model using joblib
joblib.dump(model, "model_compressed.pkl", compress=3)
print("✅ Model saved as model_compressed.pkl using joblib.")
