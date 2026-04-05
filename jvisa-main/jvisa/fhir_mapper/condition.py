"""Build FHIR Condition resources from a MIMIC-IV ICU row."""

from __future__ import annotations

from typing import Any

from .codesystems import CONDITIONS, FHIR_CONDITION_CATEGORY, ICD10, SNOMED
from .utils import codeable_concept, is_missing, make_id, reference


def build_conditions(
    row: dict[str, Any],
    patient_id: str,
    encounter_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return one Condition resource for every *active* (==1) comorbidity flag
    as well as the sepsis_label flag.
    """
    conditions: list[dict] = []

    for col, (icd10, snomed, display) in CONDITIONS.items():
        value = row.get(col)
        if is_missing(value):
            continue
        # Only create a Condition when the flag is truthy (1 / True / "1")
        if int(float(value)) != 1:
            continue

        condition: dict[str, Any] = {
            "resourceType": "Condition",
            "id": make_id(),
            "clinicalStatus": codeable_concept(
                "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "active",
                "Active",
            ),
            "verificationStatus": codeable_concept(
                "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "confirmed",
                "Confirmed",
            ),
            "category": [
                codeable_concept(
                    FHIR_CONDITION_CATEGORY,
                    "encounter-diagnosis",
                    "Encounter Diagnosis",
                )
            ],
            "code": {
                "coding": [
                    {"system": ICD10, "code": icd10, "display": display},
                    {"system": SNOMED, "code": snomed, "display": display},
                ],
                "text": display,
            },
            "subject": reference("Patient", patient_id),
        }

        if encounter_id:
            condition["encounter"] = reference("Encounter", encounter_id)

        conditions.append(condition)

    return conditions
