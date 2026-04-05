#!/usr/bin/env python3
"""
Example: convert the MIMIC-IV ICU synthetic dataset into FHIR Bundles.

Run:
    python run_fhir_mapper.py

Run in cli:
    python -m jvisa.fhir_mapper data.csv -o bundles.ndjson
    python -m jvisa.fhir_mapper data.csv -o bundles.json
"""

import json
import os

from jvisa.fhir_mapper import MIMICToFHIRMapper

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "..", "dataset", "MIMIC-IV-ICU-synthetic", "data.csv")


def main() -> None:
    mapper = MIMICToFHIRMapper()

    # --- 1. Convert the entire CSV to NDJSON (one Bundle per line) ------
    mapper.map_csv_to_ndjson(CSV_PATH, output_path="../dataset/MIMIC-IV-ICU-synthetic/bundles.ndjson")
    mapper.map_csv_to_json(CSV_PATH,   output_path="../dataset/MIMIC-IV-ICU-synthetic/bundles.json")

    # --- 2. Pretty-print the first patient's Bundle --------------------
    for bundle in mapper.iter_csv(CSV_PATH):
        print("\n── First patient FHIR Bundle (preview) ──────────────────")
        print(json.dumps(bundle, indent=2)[:3000], "...")

        # Summary
        entries = bundle.get("entry", [])
        types = {}
        for e in entries:
            rt = e["resource"]["resourceType"]
            types[rt] = types.get(rt, 0) + 1
        print("\nResource counts:", types)
        break


if __name__ == "__main__":
    main()
