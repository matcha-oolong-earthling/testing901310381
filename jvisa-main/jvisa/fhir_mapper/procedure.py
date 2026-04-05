"""Build FHIR Procedure resources from a MIMIC-IV ICU row."""

from __future__ import annotations

from typing import Any

from .codesystems import PROCEDURES, SNOMED
from .utils import codeable_concept, is_missing, make_id, quantity, reference


def build_procedures(
    row: dict[str, Any],
    patient_id: str,
    encounter_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return one Procedure per active intervention flag.

    Additional quantitative columns are attached as extensions:
      • fluids_ml_24h             → Procedure note (for fluids)
      • vasopressor_dose_mcg_kg_min → Procedure note (for vasopressors)
    """
    procedures: list[dict] = []

    for col, (snomed, display) in PROCEDURES.items():
        value = row.get(col)
        if is_missing(value):
            continue
        if int(float(value)) != 1:
            continue

        proc: dict[str, Any] = {
            "resourceType": "Procedure",
            "id": make_id(),
            "status": "completed",
            "code": codeable_concept(SNOMED, snomed, display),
            "subject": reference("Patient", patient_id),
        }

        if encounter_id:
            proc["encounter"] = reference("Encounter", encounter_id)

        # Attach dose / volume as a note when available
        if col == "vasopressors_flag" and not is_missing(
            row.get("vasopressor_dose_mcg_kg_min")
        ):
            proc["note"] = [
                {
                    "text": (
                        f"Vasopressor dose: "
                        f"{row['vasopressor_dose_mcg_kg_min']} mcg/kg/min"
                    )
                }
            ]

        procedures.append(proc)

    # --- IV Fluids (not a binary flag – continuous volume) ---------------
    if not is_missing(row.get("fluids_ml_24h")):
        fluids_vol = float(row["fluids_ml_24h"])
        if fluids_vol > 0:
            proc = {
                "resourceType": "Procedure",
                "id": make_id(),
                "status": "completed",
                "code": codeable_concept(SNOMED, "118431008", "IV fluid administration"),
                "subject": reference("Patient", patient_id),
                "note": [{"text": f"24-hour fluid volume: {fluids_vol} mL"}],
            }
            if encounter_id:
                proc["encounter"] = reference("Encounter", encounter_id)
            procedures.append(proc)

    return procedures
