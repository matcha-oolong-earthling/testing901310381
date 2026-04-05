#!/usr/bin/env python3
"""
Generate plots for the Sepsis Random Forest model report.

Produces a single figure with 4 panels:
  1. ROC curve (noise_scale=2.0)
  2. Top 15 feature importances (noise_scale=2.0)
  3. Confusion matrix heatmap (noise_scale=2.0)
  4. Performance metrics across noise levels
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

from jvisa.csv_mapper import FHIRToDataFrameMapper
from jvisa.model import SepsisRandomForest

OUTPUT_DIR = pathlib.Path("figures")
DATA_PATH = pathlib.Path("dataset/MIMIC-IV-ICU-synthetic/bundles.ndjson")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    print("Loading FHIR bundles ...")
    mapper = FHIRToDataFrameMapper()
    df = mapper.from_ndjson(DATA_PATH)
    df = mapper.impute(df, strategy="median")
    print(f"  {len(df)} encounters, sepsis prevalence {df['sepsis_label'].mean():.1%}")

    # ------------------------------------------------------------------ #
    # Panel 4 data: sweep noise levels
    # ------------------------------------------------------------------ #
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

    for ns in noise_levels:
        model = SepsisRandomForest(noise_scale=ns)
        res = model.train_and_evaluate(df)
        metrics["accuracy"].append(res.accuracy)
        metrics["precision"].append(res.precision)
        metrics["recall"].append(res.recall)
        metrics["f1"].append(res.f1)
        metrics["roc_auc"].append(res.roc_auc)
        print(f"  noise={ns:.1f}  F1={res.f1:.3f}  AUC={res.roc_auc:.3f}")

    # ------------------------------------------------------------------ #
    # Train the main model at noise_scale=2.0 for panels 1–3
    # ------------------------------------------------------------------ #
    model = SepsisRandomForest(noise_scale=2.0)
    X, y = model._prepare(df)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    model.clf.fit(X_train, y_train)
    y_pred = model.clf.predict(X_test)
    y_proba = model.clf.predict_proba(X_test)[:, 1]

    importances = dict(zip(model.feature_cols, model.clf.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]

    # ------------------------------------------------------------------ #
    # Build figure
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("SepsisAlert — Random Forest Model Results", fontsize=16, fontweight="bold")

    # --- Panel 1: ROC curve ---
    ax = axes[0, 0]
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="RF (noise=2.0)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("ROC Curve (noise_scale=2.0)")

    # --- Panel 2: Feature importances ---
    ax = axes[0, 1]
    names = [n for n, _ in reversed(sorted_imp)]
    vals = [v for _, v in reversed(sorted_imp)]
    bars = ax.barh(names, vals, color="#4C8BF5")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances")
    ax.tick_params(axis="y", labelsize=9)

    # --- Panel 3: Confusion matrix ---
    ax = axes[1, 0]
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Sepsis", "Sepsis"])
    ax.set_yticklabels(["No Sepsis", "Sepsis"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (noise_scale=2.0)")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- Panel 4: Metrics vs noise ---
    ax = axes[1, 1]
    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        label = metric_name.replace("_", " ").upper() if metric_name == "roc_auc" else metric_name.capitalize()
        ax.plot(noise_levels, metrics[metric_name], "o-", label=label, markersize=5)
    ax.set_xlabel("Noise Scale (x feature std)")
    ax.set_ylabel("Score")
    ax.set_title("Performance vs. Noise Level")
    ax.legend(fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "sepsis_rf_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
