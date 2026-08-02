"""
Single inference entry point for the smartwatch rating model.

This is the ONLY place that should ever call model.predict() in this
project outside of the training notebooks themselves.
"""

import json
from pathlib import Path
import joblib

from .feature import FeaturePipeline, ARTIFACTS_DIR

_model = None
_pipeline = None
_metadata = None


def _load():
    global _model, _pipeline, _metadata
    if _model is None:
        _model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    if _pipeline is None:
        _pipeline = FeaturePipeline(ARTIFACTS_DIR)
    if _metadata is None:
        with open(ARTIFACTS_DIR / "metadata.json") as f:
            _metadata = json.load(f)
    return _model, _pipeline, _metadata


def get_metadata() -> dict:
    """Returns the training metadata for the currently loaded production model."""
    _, _, metadata = _load()
    return metadata


def predict(raw: dict) -> dict:
    """Predicts a smartwatch rating from raw input fields.

    Args:
        raw: dict with keys Brand, Current Price, Original Price,
             Number OF Ratings, Dial Shape, Strap Color, Strap Material,
             Touchscreen, Battery Life (Days), Bluetooth, Display Size,
             Weight (optional). See feature.REQUIRED_FIELDS.

    Returns:
        {"predicted_rating": float, "model_name": str, "warnings": list}
        Raises ValueError if raw is missing required fields or has
        internally inconsistent values (e.g. Current Price > Original Price).
    """
    model, pipeline, metadata = _load()

    errors = pipeline.validate(raw)
    if errors:
        raise ValueError("; ".join(errors))

    X = pipeline.transform(raw)
    prediction = float(model.predict(X)[0])

    # Ratings are on a 1-5 scale in the training data -- clip defensively,
    # a linear model can technically predict outside that range.
    clipped = max(1.0, min(5.0, prediction))

    warnings = []
    if abs(clipped - prediction) > 1e-6:
        warnings.append(f"Raw prediction {prediction:.3f} was outside the 1-5 range and was clipped.")

    return {
        "predicted_rating": round(clipped, 3),
        "model_name": metadata["model_name"],
        "model_cv_r2": metadata["cv_r2_mean"],
        "warnings": warnings,
    }


if __name__ == "__main__":
    # Quick manual smoke test: python -m smartwatch_ml.predict
    sample = {
        "Brand": "noise",
        "Current Price": 2999,
        "Original Price": 4999,
        "Number OF Ratings": 1250,
        "Dial Shape": "Rectangle",
        "Strap Color": "Black",
        "Strap Material": "Silicon",
        "Touchscreen": "Yes",
        "Battery Life (Days)": 7,
        "Bluetooth": "Yes",
        "Display Size": 1.8,
        "Weight": "35 - 50 g",
    }
    print(predict(sample))