"""Build FHIR Observation resources from a MIMIC-IV ICU row."""

from __future__ import annotations

from typing import Any

from .codesystems import (
    BODY_MEASUREMENTS,
    FHIR_OBS_CATEGORY,
    LAB_RESULTS,
    LOINC,
    SCORES,
    UCUM,
    VITAL_SIGNS,
)
from .utils import codeable_concept, is_missing, make_id, quantity, reference


# ── internal helpers ────────────────────────────────────────────────────────

def _base_observation(
    patient_id: str,
    encounter_id: str | None,
    category_code: str,
    category_display: str,
) -> dict[str, Any]:
    """Scaffold shared by every Observation."""
    obs: dict[str, Any] = {
        "resourceType": "Observation",
        "id": make_id(),
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": FHIR_OBS_CATEGORY,
                        "code": category_code,
                        "display": category_display,
                    }
                ]
            }
        ],
        "subject": reference("Patient", patient_id),
    }
    if encounter_id:
        obs["encounter"] = reference("Encounter", encounter_id)
    return obs


# ── public API ──────────────────────────────────────────────────────────────

def build_vital_observations(
    row: dict[str, Any],
    patient_id: str,
    encounter_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return one Observation per non-null vital-sign column."""
    observations: list[dict] = []

    for col, (loinc, display, unit, ucum) in VITAL_SIGNS.items():
        value = row.get(col)
        if is_missing(value):
            continue
        obs = _base_observation(patient_id, encounter_id, "vital-signs", "Vital Signs")
        obs["code"] = codeable_concept(LOINC, loinc, display)
        obs["valueQuantity"] = quantity(float(value), unit, ucum)
        # Tag the statistic type in a component note
        if "_mean" in col:
            obs["note"] = [{"text": "Statistic: mean"}]
        elif "_max" in col:
            obs["note"] = [{"text": "Statistic: max"}]
        elif "_min" in col:
            obs["note"] = [{"text": "Statistic: min"}]
        observations.append(obs)

    return observations


def build_lab_observations(
    row: dict[str, Any],
    patient_id: str,
    encounter_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return one Observation per non-null lab-result column."""
    observations: list[dict] = []

    for col, (loinc, display, unit, ucum) in LAB_RESULTS.items():
        value = row.get(col)
        if is_missing(value):
            continue
        obs = _base_observation(patient_id, encounter_id, "laboratory", "Laboratory")
        obs["code"] = codeable_concept(LOINC, loinc, display)
        obs["valueQuantity"] = quantity(float(value), unit, ucum)
        observations.append(obs)

    return observations


def build_body_measurement_observations(
    row: dict[str, Any],
    patient_id: str,
    encounter_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return Observations for weight, height, and BMI."""
    observations: list[dict] = []

    for col, (loinc, display, unit, ucum) in BODY_MEASUREMENTS.items():
        value = row.get(col)
        if is_missing(value):
            continue
        obs = _base_observation(patient_id, encounter_id, "vital-signs", "Vital Signs")
        obs["code"] = codeable_concept(LOINC, loinc, display)
        obs["valueQuantity"] = quantity(float(value), unit, ucum)
        observations.append(obs)

    return observations


def build_score_observations(
    row: dict[str, Any],
    patient_id: str,
    encounter_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return Observations for clinical scores (SOFA, APACHE IV, etc.)."""
    observations: list[dict] = []

    for col, (loinc, display) in SCORES.items():
        value = row.get(col)
        if is_missing(value):
            continue
        obs = _base_observation(patient_id, encounter_id, "survey", "Survey")
        obs["code"] = codeable_concept(LOINC, loinc, display)

        # FiO2 is a percentage; others are integer scores
        if col == "fio2_percent":
            obs["valueQuantity"] = quantity(float(value), "%", "%")
        else:
            obs["valueInteger"] = int(float(value))

        observations.append(obs)

    return observations
