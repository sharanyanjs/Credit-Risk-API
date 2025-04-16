import joblib
from sklearn.ensemble import RandomForestClassifier
from app.core.config import settings
from pathlib import Path

MODEL_PATH = Path(settings.MODEL_DIR) / "credit_risk_model.pkl"

def load_model():
    """Load trained model from disk"""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def train_model(X, y):
    """Train and save new model"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X, y)
    
    # Save model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    return model
