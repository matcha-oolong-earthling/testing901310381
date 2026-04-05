"""Build a FHIR Patient resource from a MIMIC-IV ICU row."""

from __future__ import annotations

from typing import Any

from .codesystems import (
    ETHNICITY_MAP,
    GENDER_MAP,
    V3_ADMIN_GENDER,
    V3_ETHNICITY,
)
from .utils import coding, is_missing, make_id


def build_patient(row: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single CSV row into a FHIR R4 Patient resource.

    Mapped fields
    --------------
    subject_id  → Patient.identifier
    gender      → Patient.gender
    age         → Patient.extension (approx birth year)
    ethnicity   → Patient.extension (us-core-race)
    insurance   → Patient.extension (custom)
    """
    patient_id = str(row["subject_id"])

    resource: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [
            {
                "system": "http://mimic.mit.edu/fhir/mimic/identifier/patient",
                "value": patient_id,
            }
        ],
    }

    # --- gender ---------------------------------------------------------
    raw_gender = str(row.get("gender", "U")).strip().upper()
    resource["gender"] = GENDER_MAP.get(raw_gender, "unknown")

    # --- approximate birth year from age --------------------------------
    if not is_missing(row.get("age")):
        from datetime import date

        approx_birth_year = date.today().year - int(float(row["age"]))
        resource["birthDate"] = f"{approx_birth_year}"

    # --- extensions (race / ethnicity / insurance) ----------------------
    extensions: list[dict] = []

    eth = str(row.get("ethnicity", "")).strip()
    if eth and eth in ETHNICITY_MAP:
        code, display = ETHNICITY_MAP[eth]
        extensions.append(
            {
                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                "extension": [
                    {
                        "url": "ombCategory",
                        "valueCoding": coding(V3_ETHNICITY, code, display),
                    },
                    {"url": "text", "valueString": display},
                ],
            }
        )

    insurance = str(row.get("insurance", "")).strip()
    if insurance:
        extensions.append(
            {
                "url": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/insurance",
                "valueString": insurance,
            }
        )

    if extensions:
        resource["extension"] = extensions

    return resource
