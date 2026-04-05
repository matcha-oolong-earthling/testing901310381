"""
CLI entry-point for jvisa.fhir_mapper.

Usage
-----
    python -m jvisa.fhir_mapper data.csv -o bundles.ndjson
    python -m jvisa.fhir_mapper data.csv -o bundles.json
"""

from __future__ import annotations

import argparse
import os

from .mapper import MIMICToFHIRMapper


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="jvisa.fhir_mapper",
        description="Convert MIMIC-IV ICU CSV data into FHIR R4 Bundles.",
    )
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path. Format is inferred from extension (.ndjson or .json).",
    )
    parser.add_argument(
        "-b",
        "--bundle-type",
        choices=["collection", "transaction", "batch"],
        default="collection",
        help="FHIR Bundle type. Default: collection",
    )

    args = parser.parse_args(argv)
    mapper = MIMICToFHIRMapper()

    _, ext = os.path.splitext(args.output)
    ext = ext.lower()

    if ext == ".json":
        mapper.map_csv_to_json(args.csv, args.output, bundle_type=args.bundle_type)
    elif ext == ".ndjson":
        mapper.map_csv_to_ndjson(args.csv, args.output, bundle_type=args.bundle_type)
    else:
        parser.error(f"Unsupported extension '{ext}'. Use .ndjson or .json.")


if __name__ == "__main__":
    main()
