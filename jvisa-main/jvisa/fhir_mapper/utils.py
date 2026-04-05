"""Utility helpers shared across resource builders."""

from __future__ import annotations

import math
import uuid
from typing import Any


def make_id() -> str:
    """Generate a deterministic-friendly UUID string."""
    return str(uuid.uuid4())


def is_missing(value: Any) -> bool:
    """Return True if *value* is NaN, None, or an empty string."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def coding(system: str, code: str, display: str | None = None) -> dict:
    """Build a FHIR Coding element."""
    c: dict[str, str] = {"system": system, "code": str(code)}
    if display:
        c["display"] = display
    return c


def codeable_concept(
    system: str, code: str, display: str | None = None, text: str | None = None
) -> dict:
    """Build a FHIR CodeableConcept element."""
    cc: dict[str, Any] = {"coding": [coding(system, code, display)]}
    if text:
        cc["text"] = text
    return cc


def quantity(value: float, unit: str, ucum_unit: str | None = None) -> dict:
    """Build a FHIR Quantity element."""
    q: dict[str, Any] = {"value": round(value, 4), "unit": unit}
    if ucum_unit:
        q["system"] = "http://unitsofmeasure.org"
        q["code"] = ucum_unit
    return q


def reference(resource_type: str, resource_id: str) -> dict:
    """Build a FHIR Reference element."""
    return {"reference": f"{resource_type}/{resource_id}"}
