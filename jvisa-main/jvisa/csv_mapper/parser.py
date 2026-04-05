"""
Reverse-map FHIR R4 Bundles (produced by jvisa.fhir_mapper) back into
flat tabular rows suitable for a pandas DataFrame.
"""

from __future__ import annotations

import json
import pathlib
import re
from datetime import date
from typing import Any

import pandas as pd

from jvisa.fhir_mapper.codesystems import (
    BODY_MEASUREMENTS,
    CONDITIONS,
    LAB_RESULTS,
    PROCEDURES,
    SCORES,
    VITAL_SIGNS,
)

# ---------------------------------------------------------------------------
# Build reverse-lookup dicts: FHIR display/code → original CSV column name
# ---------------------------------------------------------------------------

# Observations: display string → column name
_OBS_DISPLAY_TO_COL: dict[str, str] = {}
for _col, (_loinc, _display, *_rest) in {
    **VITAL_SIGNS,
    **LAB_RESULTS,
    **BODY_MEASUREMENTS,
    **SCORES,
}.items():
    _OBS_DISPLAY_TO_COL[_display] = _col

# Conditions: SNOMED code → column name
_COND_SNOMED_TO_COL: dict[str, str] = {
    snomed: col for col, (_icd, snomed, _disp) in CONDITIONS.items()
}

# Procedures: SNOMED code → column name
_PROC_SNOMED_TO_COL: dict[str, str] = {
    snomed: col for col, (snomed, _disp) in PROCEDURES.items()
}

# Admit-source code → original string
_ADMIT_SOURCE_REVERSE: dict[str, str] = {
    "emd": "ED",
    "outp": "OR",
    "hosp-trans": "Transfer",
}


# ---------------------------------------------------------------------------
# Bundle → flat row
# ---------------------------------------------------------------------------

def _parse_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Extract a single flat dict from one FHIR Bundle."""
    row: dict[str, Any] = {}

    # Collect all condition/procedure columns so we can default absent ones to 0
    all_condition_cols = set(CONDITIONS.keys())
    all_procedure_cols = set(PROCEDURES.keys())
    seen_conditions: set[str] = set()
    seen_procedures: set[str] = set()

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")

        if rtype == "Patient":
            _parse_patient(resource, row)
        elif rtype == "Encounter":
            _parse_encounter(resource, row)
        elif rtype == "Observation":
            _parse_observation(resource, row)
        elif rtype == "Condition":
            _parse_condition(resource, row, seen_conditions)
        elif rtype == "Procedure":
            _parse_procedure(resource, row, seen_procedures)

    # Binary flags: 0 for absent conditions/procedures
    for col in all_condition_cols:
        if col not in seen_conditions:
            row[col] = 0
    for col in all_procedure_cols:
        if col not in seen_procedures:
            row[col] = 0

    return row


def _parse_patient(resource: dict, row: dict) -> None:
    row["subject_id"] = resource.get("id")

    # Gender → M/F/O/U
    gender = resource.get("gender", "unknown")
    gender_reverse = {"male": "M", "female": "F", "other": "O", "unknown": "U"}
    row["gender"] = gender_reverse.get(gender, "U")

    # Age from birthDate (approximate birth year)
    birth_date = resource.get("birthDate")
    if birth_date:
        birth_year = int(birth_date[:4])
        row["age"] = date.today().year - birth_year

    # Extensions: ethnicity, insurance
    for ext in resource.get("extension", []):
        url = ext.get("url", "")
        if "us-core-race" in url:
            for sub in ext.get("extension", []):
                if sub.get("url") == "text":
                    row["ethnicity"] = sub.get("valueString")
        elif "insurance" in url:
            row["insurance"] = ext.get("valueString")


def _parse_encounter(resource: dict, row: dict) -> None:
    # Length of stay
    length = resource.get("length")
    if length:
        row["icu_los_hours"] = length.get("value")

    # Admit source
    hosp = resource.get("hospitalization", {})
    admit_src = hosp.get("admitSource", {})
    codings = admit_src.get("coding", [])
    if codings:
        code = codings[0].get("code", "")
        row["hospital_admit_source"] = _ADMIT_SOURCE_REVERSE.get(code, code)

    # Extensions
    for ext in resource.get("extension", []):
        url = ext.get("url", "")
        if "icu-admit-hour" in url:
            row["icu_admit_time_hour"] = ext.get("valueInteger")
        elif "day-of-week" in url:
            row["day_of_week"] = ext.get("valueInteger")
        elif "readmission-30day" in url:
            row["readmission_30day"] = int(ext.get("valueBoolean", False))


def _parse_observation(resource: dict, row: dict) -> None:
    codings = resource.get("code", {}).get("coding", [])
    if not codings:
        return
    display = codings[0].get("display", "")
    col = _OBS_DISPLAY_TO_COL.get(display)
    if col is None:
        return

    # Extract value — either valueQuantity, valueInteger, or valueString
    if "valueQuantity" in resource:
        row[col] = resource["valueQuantity"]["value"]
    elif "valueInteger" in resource:
        row[col] = resource["valueInteger"]


def _parse_condition(resource: dict, row: dict, seen: set) -> None:
    codings = resource.get("code", {}).get("coding", [])
    for coding in codings:
        snomed = coding.get("code", "")
        col = _COND_SNOMED_TO_COL.get(snomed)
        if col:
            row[col] = 1
            seen.add(col)
            return


def _parse_procedure(resource: dict, row: dict, seen: set) -> None:
    codings = resource.get("code", {}).get("coding", [])
    if not codings:
        return
    snomed = codings[0].get("code", "")
    col = _PROC_SNOMED_TO_COL.get(snomed)

    if col:
        row[col] = 1
        seen.add(col)

        # Extract vasopressor dose from note
        if col == "vasopressors_flag":
            for note in resource.get("note", []):
                m = re.search(r"dose:\s*([\d.]+)", note.get("text", ""))
                if m:
                    row["vasopressor_dose_mcg_kg_min"] = float(m.group(1))

    # IV fluids (special: mapped to SNOMED 118431008, not a binary flag column)
    if snomed == "118431008":
        for note in resource.get("note", []):
            m = re.search(r"volume:\s*([\d.]+)", note.get("text", ""))
            if m:
                row["fluids_ml_24h"] = float(m.group(1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_IMPUTE_STRATEGIES = {"median", "mean", "zero"}


class FHIRToDataFrameMapper:
    """Convert FHIR Bundles (JSON / NDJSON) back into a pandas DataFrame."""

    def parse_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """Parse a single FHIR Bundle dict into a flat row dict."""
        return _parse_bundle(bundle)

    def from_bundles(self, bundles: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of FHIR Bundle dicts into a DataFrame."""
        rows = [_parse_bundle(b) for b in bundles]
        return pd.DataFrame(rows)

    def from_json(self, path: str | pathlib.Path) -> pd.DataFrame:
        """Read a JSON array of Bundles and return a DataFrame."""
        path = pathlib.Path(path)
        with open(path, encoding="utf-8") as fh:
            bundles = json.load(fh)
        return self.from_bundles(bundles)

    def from_ndjson(self, path: str | pathlib.Path) -> pd.DataFrame:
        """Read an NDJSON file (one Bundle per line) and return a DataFrame."""
        path = pathlib.Path(path)
        bundles = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    bundles.append(json.loads(line))
        return self.from_bundles(bundles)

    # ------------------------------------------------------------------ #
    #  Imputation                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def impute(
        df: pd.DataFrame,
        strategy: str = "median",
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fill missing (NaN) values in numeric columns.

        Parameters
        ----------
        df : DataFrame
            Input data (not modified in place).
        strategy : {"median", "mean", "zero"}
            How to fill missing values in numeric columns.
        columns : list[str] or None
            Specific columns to impute. If None, all numeric columns
            with at least one NaN are imputed.

        Returns
        -------
        DataFrame with NaN values filled.
        """
        if strategy not in _IMPUTE_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from {sorted(_IMPUTE_STRATEGIES)}."
            )

        df = df.copy()
        numeric_cols = df.select_dtypes(include="number").columns
        if columns is not None:
            numeric_cols = numeric_cols.intersection(columns)

        if strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "zero":
            df[numeric_cols] = df[numeric_cols].fillna(0)

        return df
