"""Build a FHIR Encounter resource from a MIMIC-IV ICU row."""

from __future__ import annotations

from typing import Any

from .codesystems import ENCOUNTER_CLASS_SYSTEM, SNOMED
from .utils import codeable_concept, is_missing, make_id, quantity, reference


def build_encounter(
    row: dict[str, Any],
    patient_id: str,
) -> dict[str, Any]:
    """
    Map ICU-stay metadata to a FHIR R4 Encounter resource.

    Mapped fields
    --------------
    icu_los_hours          → Encounter.length
    hospital_admit_source  → Encounter.hospitalization.admitSource
    icu_admit_time_hour    → Encounter.extension (custom, hour of day)
    day_of_week            → Encounter.extension (custom)
    readmission_30day      → Encounter.extension (custom flag)
    """
    encounter_id = make_id()

    encounter: dict[str, Any] = {
        "resourceType": "Encounter",
        "id": encounter_id,
        "status": "finished",
        "class": {
            "system": ENCOUNTER_CLASS_SYSTEM,
            "code": "IMP",
            "display": "inpatient encounter",
        },
        "type": [
            codeable_concept(SNOMED, "305351004", "ICU admission")
        ],
        "subject": reference("Patient", patient_id),
    }

    # --- length of stay --------------------------------------------------
    if not is_missing(row.get("icu_los_hours")):
        encounter["length"] = {
            "value": round(float(row["icu_los_hours"]), 2),
            "unit": "hours",
            "system": "http://unitsofmeasure.org",
            "code": "h",
        }

    # --- admit source ----------------------------------------------------
    admit_source = str(row.get("hospital_admit_source", "")).strip()
    if admit_source:
        source_map: dict[str, tuple[str, str]] = {
            "ED":       ("emd",      "From accident/emergency department"),
            "OR":       ("outp",     "From outpatient department (OR)"),
            "Transfer": ("hosp-trans", "Transferred from other hospital"),
        }
        code, display = source_map.get(admit_source, ("other", admit_source))
        encounter["hospitalization"] = {
            "admitSource": codeable_concept(
                "http://terminology.hl7.org/CodeSystem/admit-source",
                code,
                display,
            )
        }

    # --- extensions (admit hour, day of week, readmission) ---------------
    extensions: list[dict] = []

    if not is_missing(row.get("icu_admit_time_hour")):
        extensions.append(
            {
                "url": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/icu-admit-hour",
                "valueInteger": int(float(row["icu_admit_time_hour"])),
            }
        )

    if not is_missing(row.get("day_of_week")):
        extensions.append(
            {
                "url": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/day-of-week",
                "valueInteger": int(float(row["day_of_week"])),
            }
        )

    if not is_missing(row.get("readmission_30day")):
        extensions.append(
            {
                "url": "http://mimic.mit.edu/fhir/mimic/StructureDefinition/readmission-30day",
                "valueBoolean": int(float(row["readmission_30day"])) == 1,
            }
        )

    if extensions:
        encounter["extension"] = extensions

    return encounter
