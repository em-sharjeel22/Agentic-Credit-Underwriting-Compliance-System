# create_dummy_model.py
# Creates a tiny XGBoost model bundle compatible with the app's expectations
# Saves the bundle to models/credit_model.pkl

import joblib
import pathlib
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

# Create a small synthetic dataset with 29 features (the app expects ~29 features)
X, y = make_classification(n_samples=300, n_features=29, n_informative=10, random_state=42)

# Train a compact XGBoost booster
dtrain = xgb.DMatrix(X, label=y)
params = {"objective": "binary:logistic", "verbosity": 0}
bst = xgb.train(params, dtrain, num_boost_round=20)

# Wrapper so the object exposes predict_proba and get_booster()
class XGBWrapper:
    def __init__(self, booster, feature_count):
        self._booster = booster
        # create placeholder feature names matching the app's expected order
        self._feature_names = [f"f{i}" for i in range(feature_count)]

    def predict_proba(self, X_df):
        # Accept either DataFrame-like or numpy array
        if hasattr(X_df, "values"):
            arr = X_df.values
        else:
            arr = np.asarray(X_df)
        d = xgb.DMatrix(arr)
        p = self._booster.predict(d)
        # return shape (n_samples, 2)
        return np.vstack([1 - p, p]).T

    def get_booster(self):
        return self._booster

    @property
    def feature_names(self):
        return self._feature_names

# Build bundle expected by the app: {"model": model, "threshold": float}
model = XGBWrapper(bst, feature_count=29)
bundle = {"model": model, "threshold": 0.5}

# Ensure models directory exists and save
out_dir = pathlib.Path("models")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "credit_model.pkl"
joblib.dump(bundle, out_path)

print(f"Saved dummy model bundle to {out_path.resolve()}")
