#!/usr/bin/env python3
"""
End-to-end pipeline: FHIR Bundles → DataFrame → Random Forest sepsis prediction.

Usage
-----
    python scripts/run_sepsis_model.py
    python scripts/run_sepsis_model.py --input dataset/MIMIC-IV-ICU-synthetic/bundles.json
    python scripts/run_sepsis_model.py --input dataset/MIMIC-IV-ICU-synthetic/bundles.ndjson
"""

from __future__ import annotations

import argparse
import pathlib

from jvisa.csv_mapper import FHIRToDataFrameMapper
from jvisa.model import SepsisRandomForest

DEFAULT_INPUT = pathlib.Path("dataset/MIMIC-IV-ICU-synthetic/bundles.ndjson")


def main() -> None:
    parser = argparse.ArgumentParser(description="FHIR → DataFrame → Sepsis RF model")
    parser.add_argument(
        "--input", "-i",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help="Path to bundles.json or bundles.ndjson",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument(
        "--noise-scale", type=float, default=0.0,
        help="Gaussian noise scale (relative to each feature's std). 0 = no noise.",
    )
    parser.add_argument(
        "--impute", choices=["median", "mean", "zero"], default="median",
        help="Imputation strategy for missing values (default: median)",
    )
    args = parser.parse_args()

    # Step 1: FHIR → DataFrame
    print(f"Loading FHIR bundles from {args.input} ...")
    mapper = FHIRToDataFrameMapper()
    if args.input.suffix == ".ndjson":
        df = mapper.from_ndjson(args.input)
    else:
        df = mapper.from_json(args.input)
    print(f"Parsed {len(df)} patient encounters, {len(df.columns)} columns")

    n_missing = df.select_dtypes(include="number").isna().sum().sum()
    print(f"Missing numeric values: {n_missing}")

    # Step 2: Impute missing values
    df = mapper.impute(df, strategy=args.impute)
    print(f"Imputed with strategy: {args.impute}")
    print(f"Sepsis prevalence: {df['sepsis_label'].mean():.1%}")
    if args.noise_scale > 0:
        print(f"Noise scale: {args.noise_scale}")
    print()

    # Step 3: Train & evaluate Random Forest
    model = SepsisRandomForest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        noise_scale=args.noise_scale,
    )
    results = model.train_and_evaluate(df)
    print(results.summary())


if __name__ == "__main__":
    main()
