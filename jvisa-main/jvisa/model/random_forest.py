"""
Random-forest classifier for ICU sepsis prediction.

Consumes a pandas DataFrame (produced by :class:`jvisa.csv_mapper.FHIRToDataFrameMapper`)
and trains a scikit-learn ``RandomForestClassifier`` on clinical features to
predict the ``sepsis_label`` column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Columns that should NOT be used as features
_NON_FEATURE_COLS = {
    "subject_id",
    "sepsis_label",        # target
    "ethnicity",           # categorical string — not one-hot encoded here
    "insurance",           # categorical string
    "gender",              # will be encoded separately
    "hospital_admit_source",  # categorical string
}

TARGET_COL = "sepsis_label"


@dataclass
class EvalResults:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: np.ndarray
    classification_report: str
    feature_importances: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Sepsis Random Forest — Evaluation ===",
            f"Accuracy:  {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1:        {self.f1:.4f}",
        ]
        if self.roc_auc is not None:
            lines.append(f"ROC AUC:   {self.roc_auc:.4f}")
        lines.append("")
        lines.append("Confusion matrix:")
        lines.append(str(self.confusion_matrix))
        lines.append("")
        lines.append(self.classification_report)

        if self.feature_importances:
            lines.append("Top 15 feature importances:")
            sorted_fi = sorted(
                self.feature_importances.items(), key=lambda x: x[1], reverse=True
            )
            for name, imp in sorted_fi[:15]:
                lines.append(f"  {name:35s} {imp:.4f}")

        return "\n".join(lines)


class SepsisRandomForest:
    """Train and evaluate a random-forest model for sepsis prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
        test_size: float = 0.2,
        noise_scale: float = 0.0,
        **rf_kwargs: Any,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.noise_scale = noise_scale
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
            **rf_kwargs,
        )
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------ #
    #  Preprocessing                                                      #
    # ------------------------------------------------------------------ #

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Select features, encode categoricals, handle missing values."""
        df = df.copy()

        # Encode gender as binary (M=1, F=0, other/unknown=NaN)
        if "gender" in df.columns:
            df["gender_male"] = (df["gender"] == "M").astype(float)
            df.loc[~df["gender"].isin(["M", "F"]), "gender_male"] = np.nan

        # Encode hospital_admit_source as dummies
        if "hospital_admit_source" in df.columns:
            dummies = pd.get_dummies(
                df["hospital_admit_source"], prefix="admit_src"
            ).astype(float)
            df = pd.concat([df, dummies], axis=1)

        # Select numeric feature columns
        self.feature_cols = [
            c
            for c in df.columns
            if c not in _NON_FEATURE_COLS and df[c].dtype in [np.float64, np.int64, float, int]
        ]

        X = df[self.feature_cols].copy()
        y = df[TARGET_COL].astype(int)

        # Fill missing values with column median
        X = X.fillna(X.median())

        # Inject Gaussian noise (scale is relative to each column's std)
        if self.noise_scale > 0:
            rng = np.random.default_rng(self.random_state)
            stds = X.std()
            noise = pd.DataFrame(
                rng.normal(0, 1, size=X.shape),
                index=X.index,
                columns=X.columns,
            )
            X = X + noise * stds * self.noise_scale

        return X, y

    # ------------------------------------------------------------------ #
    #  Train + evaluate                                                   #
    # ------------------------------------------------------------------ #

    def train_and_evaluate(self, df: pd.DataFrame) -> EvalResults:
        """
        End-to-end: preprocess → train/test split → fit → evaluate.

        Returns an :class:`EvalResults` with all metrics.
        """
        X, y = self._prepare(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y,
        )

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        y_proba = self.clf.predict_proba(X_test)

        # ROC AUC (needs probability of the positive class)
        roc_auc = None
        if y_proba.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])

        importances = dict(zip(self.feature_cols, self.clf.feature_importances_))

        return EvalResults(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc,
            confusion_matrix=confusion_matrix(y_test, y_pred),
            classification_report=classification_report(y_test, y_pred, zero_division=0),
            feature_importances=importances,
        )

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering used in _prepare, then select feature_cols."""
        df = df.copy()
        if "gender" in df.columns:
            df["gender_male"] = (df["gender"] == "M").astype(float)
            df.loc[~df["gender"].isin(["M", "F"]), "gender_male"] = np.nan
        if "hospital_admit_source" in df.columns:
            dummies = pd.get_dummies(
                df["hospital_admit_source"], prefix="admit_src"
            ).astype(float)
            df = pd.concat([df, dummies], axis=1)
        # Add any missing feature columns (e.g. an admit_src category not in this subset)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_cols].copy()
        X = X.fillna(X.median())
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict sepsis labels for new data (must call train_and_evaluate first)."""
        return self.clf.predict(self._encode(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict sepsis probabilities for new data."""
        return self.clf.predict_proba(self._encode(df))
