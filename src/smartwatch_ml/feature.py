"""
Feature transformation for the smartwatch rating model.

This module is the single source of truth for turning a raw smartwatch
description into the exact feature vector the trained model expects.
Both the notebooks (during training) and the API (at inference time)
should ultimately agree with this logic -- this file exists so there
is only one place that logic can drift.

It works entirely off the artifacts saved by 03_preprocessing:
encoder.joblib, scaler.joblib, weight_encoder.joblib, the frequency
maps, and feature_order.joblib. If those files are missing, this
module cannot function -- that's intentional, it means training
hasn't produced a usable artifact set yet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# Categories the model was actually trained on. These come from the
# grouped/cleaned training data, not the raw dataset -- used to validate
# and to power dropdown choices in a frontend.
DIAL_SHAPE_CHOICES = ["Circle", "Rectangle", "Square", "Missing", "Other"]
STRAP_MATERIAL_CHOICES = ["Silicon", "Rubber", "Stainless Steel", "Missing", "Other"]
WEIGHT_CHOICES = ["<= 20 g", "20 - 35 g", "35 - 50 g", "50 - 75 g", "75g +"]

REQUIRED_FIELDS = [
    "Brand", "Current Price", "Original Price",
    "Dial Shape", "Strap Color", "Strap Material", "Touchscreen",
    "Battery Life (Days)", "Bluetooth", "Display Size",
]
# "Number OF Ratings" and "Weight" are intentionally not required --
# a brand-new / not-yet-launched product has no review count yet,
# and Weight defaults to the most common bin if unknown.


def _sanitize_columns(cols: pd.Index) -> pd.Index:
    return (
        cols.str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            .str.replace("<", "less than", regex=False)
            .str.replace(">", "greater than", regex=False)
            .str.replace("=", "equal to", regex=False)
    )


class FeaturePipeline:
    """Loads every artifact once, then transforms raw records repeatedly.

    Usage:
        pipeline = FeaturePipeline()
        X = pipeline.transform(raw_dict)   # -> ready for model.predict(X)
    """

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir

        self.encoder = joblib.load(artifacts_dir / "encoder.joblib")
        self.scaler = joblib.load(artifacts_dir / "scaler.joblib")
        self.weight_encoder = joblib.load(artifacts_dir / "weight_encoder.joblib")
        self.feature_order = joblib.load(artifacts_dir / "feature_order.joblib")
        self.categorical_columns = joblib.load(artifacts_dir / "categorical_columns.joblib")
        self.numerical_columns = joblib.load(artifacts_dir / "numerical_columns.joblib")
        self.binary_columns = joblib.load(artifacts_dir / "binary_columns.joblib")
        self.brand_freq_map = joblib.load(artifacts_dir / "brand_freq_map.joblib")
        self.strap_color_freq_map = joblib.load(artifacts_dir / "strap_color_freq_map.joblib")
        self.default_brand_freq = joblib.load(artifacts_dir / "default_brand_freq.joblib")
        self.default_strap_color_freq = joblib.load(artifacts_dir / "default_strap_color_freq.joblib")

        weight_mode_path = artifacts_dir / "weight_mode.joblib"
        self.weight_mode = joblib.load(weight_mode_path) if weight_mode_path.exists() else "35 - 50 g"

    def validate(self, raw: dict) -> list:
        """Returns a list of human-readable problems, empty if the input is usable."""
        errors = []
        for field in REQUIRED_FIELDS:
            if field not in raw or raw[field] in (None, ""):
                if field == "Weight":
                    continue  # Weight is allowed to be missing, handled below
                errors.append(f"Missing required field: {field}")

        if raw.get("Original Price") and raw.get("Current Price"):
            if raw["Current Price"] > raw["Original Price"]:
                errors.append("Current Price cannot be greater than Original Price")

        return errors

    def transform(self, raw: dict) -> pd.DataFrame:
        """Transforms one raw smartwatch record into a single-row DataFrame
        matching the exact columns and order the model was trained on."""

        row = dict(raw)  # don't mutate caller's dict

        # --- Weight: ordinal encode, track missingness like training did ---
        weight_value = row.get("Weight")
        weight_missing = 1 if not weight_value else 0
        if not weight_value:
            weight_value = self.weight_mode

        # --- Number of ratings: log1p, same as training ---
        num_ratings = float(row.get("Number OF Ratings", 0))
        reviews_placeholder_flag = bool(num_ratings == 996)
        num_ratings_log = np.log1p(num_ratings)

        # --- Discount percentage: derive if not supplied ---
        current_price = float(row["Current Price"])
        original_price = float(row["Original Price"])
        discount_pct = row.get("Discount Percentage")
        if discount_pct is None:
            discount_pct = ((original_price - current_price) / original_price) * 100

        # --- Touchscreen / Bluetooth: accept Yes/No or 0/1 ---
        def to_binary(value):
            if isinstance(value, str):
                return 1 if value.strip().lower() == "yes" else 0
            return int(value)

        touchscreen = to_binary(row["Touchscreen"])
        bluetooth = to_binary(row["Bluetooth"])

        # --- Brand / Strap Color: frequency encode ---
        brand_freq = self.brand_freq_map.get(row["Brand"], self.default_brand_freq)
        strap_color_freq = self.strap_color_freq_map.get(row["Strap Color"], self.default_strap_color_freq)

        # --- Assemble the pre-encoding row ---
        pre = pd.DataFrame([{
            "Current Price": current_price,
            "Original Price": original_price,
            "Discount Percentage": discount_pct,
            "Number OF Ratings": num_ratings_log,
            "Dial Shape": row["Dial Shape"],
            "Strap Material": row["Strap Material"],
            "Touchscreen": touchscreen,
            "Battery Life (Days)": float(row["Battery Life (Days)"]),
            "Bluetooth": bluetooth,
            "Display Size": float(row["Display Size"]),
            "Weight_missing": weight_missing,
            "reviews_placeholder_flag": reviews_placeholder_flag,
            "Brand_freq": brand_freq,
            "Strap_Color_freq": strap_color_freq,
            "Weight": weight_value,
        }])

        # --- Weight ordinal ---
        pre["Weight_ordinal"] = self.weight_encoder.transform(pre[["Weight"]])
        pre = pre.drop(columns=["Weight"])

        # --- One-hot encode Dial Shape / Strap Material ---
        cat_encoded = self.encoder.transform(pre[self.categorical_columns])
        cat_col_names = self.encoder.get_feature_names_out(self.categorical_columns)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_col_names, index=pre.index)

        combined = pd.concat(
            [pre[self.numerical_columns], pre[self.binary_columns], cat_df],
            axis=1,
        )
        combined.columns = _sanitize_columns(combined.columns)

        # --- Reindex to the exact training column order, fill anything missing with 0 ---
        combined = combined.reindex(columns=self.feature_order, fill_value=0)

        # --- Scale (only matters for models that need it, harmless otherwise) ---
        scaled = pd.DataFrame(
            self.scaler.transform(combined),
            columns=combined.columns,
            index=combined.index,
        )

        return scaled