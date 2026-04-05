"""
Main mapper — converts a MIMIC-IV ICU row (or a full DataFrame) into
FHIR R4 resources wrapped in Bundles.
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Iterator

from .bundle import make_bundle
from .condition import build_conditions
from .encounter import build_encounter
from .observation import (
    build_body_measurement_observations,
    build_lab_observations,
    build_score_observations,
    build_vital_observations,
)
from .patient import build_patient
from .procedure import build_procedures


class MIMICToFHIRMapper:
    """
    High-level interface that turns tabular ICU data into FHIR Bundles.

    Usage
    -----
    >>> mapper = MIMICToFHIRMapper()
    >>> bundle = mapper.map_row(row_dict)           # single row → Bundle
    >>> bundles = mapper.map_csv("data.csv")         # whole CSV  → list[Bundle]
    >>> mapper.map_csv_to_ndjson("data.csv", "out/bundles.ndjson") # CSV → NDJSON file
    """

    # ------------------------------------------------------------------ #
    #  Single-row mapping                                                 #
    # ------------------------------------------------------------------ #
    def map_row(
        self,
        row: dict[str, Any],
        bundle_type: str = "collection",
    ) -> dict[str, Any]:
        """
        Convert one data row into a FHIR Bundle containing:

        * 1 Patient
        * 1 Encounter
        * N Observations  (vitals, labs, body measurements, clinical scores)
        * N Conditions     (comorbidities + sepsis label)
        * N Procedures     (interventions)
        """
        resources: list[dict[str, Any]] = []

        # 1. Patient
        patient = build_patient(row)
        patient_id = patient["id"]
        resources.append(patient)

        # 2. Encounter
        encounter = build_encounter(row, patient_id)
        encounter_id = encounter["id"]
        resources.append(encounter)

        # 3. Observations
        resources.extend(
            build_vital_observations(row, patient_id, encounter_id)
        )
        resources.extend(
            build_lab_observations(row, patient_id, encounter_id)
        )
        resources.extend(
            build_body_measurement_observations(row, patient_id, encounter_id)
        )
        resources.extend(
            build_score_observations(row, patient_id, encounter_id)
        )

        # 4. Conditions
        resources.extend(build_conditions(row, patient_id, encounter_id))

        # 5. Procedures
        resources.extend(build_procedures(row, patient_id, encounter_id))

        return make_bundle(resources, bundle_type=bundle_type)

    # ------------------------------------------------------------------ #
    #  CSV helpers                                                        #
    # ------------------------------------------------------------------ #
    def iter_csv(
        self,
        csv_path: str | pathlib.Path,
        bundle_type: str = "collection",
    ) -> Iterator[dict[str, Any]]:
        """Yield one FHIR Bundle per CSV row (lazy / streaming)."""
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield self.map_row(row, bundle_type=bundle_type)

    def map_csv(
        self,
        csv_path: str | pathlib.Path,
        bundle_type: str = "collection",
    ) -> list[dict[str, Any]]:
        """Load an entire CSV and return a list of FHIR Bundles."""
        return list(self.iter_csv(csv_path, bundle_type=bundle_type))

    def map_csv_to_ndjson(
        self,
        csv_path: str | pathlib.Path,
        output_path: str | pathlib.Path,
        bundle_type: str = "collection",
    ) -> pathlib.Path:
        """
        Stream a CSV → one NDJSON file (one Bundle-JSON per line).

        Returns the path to the written file.
        """
        output_path = pathlib.Path(output_path)
        count = 0
        with open(output_path, "w", encoding="utf-8") as fh:
            for bundle in self.iter_csv(csv_path, bundle_type=bundle_type):
                fh.write(json.dumps(bundle, separators=(",", ":")) + "\n")
                count += 1

        print(f"✅ Wrote {count} FHIR Bundles → {output_path}")
        return output_path

    def map_csv_to_json(
        self,
        csv_path: str | pathlib.Path,
        output_path: str | pathlib.Path,
        bundle_type: str = "collection",
    ) -> pathlib.Path:
        """
        Convert a full CSV into a single JSON array of FHIR Bundles.

        Returns the path to the written file.
        """
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bundles = self.map_csv(csv_path, bundle_type=bundle_type)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(bundles, fh, indent=2)

        print(f"✅ Wrote {len(bundles)} FHIR Bundles → {output_path}")
        return output_path

    # ------------------------------------------------------------------ #
    #  DataFrame support (optional pandas dependency)                      #
    # ------------------------------------------------------------------ #
    def map_dataframe(self, df: Any, bundle_type: str = "collection") -> list[dict]:
        """
        Accept a ``pandas.DataFrame`` and return a list of FHIR Bundles.

        Pandas is imported lazily so the module works without it.
        """
        bundles = []
        for _, row in df.iterrows():
            bundles.append(self.map_row(row.to_dict(), bundle_type=bundle_type))
        return bundles
