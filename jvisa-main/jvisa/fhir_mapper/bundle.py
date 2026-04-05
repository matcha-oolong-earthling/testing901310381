"""Assemble individual FHIR resources into a FHIR Bundle."""

from __future__ import annotations

from typing import Any

from .utils import make_id


def make_bundle(
    resources: list[dict[str, Any]],
    bundle_type: str = "collection",
) -> dict[str, Any]:
    """
    Wrap a list of FHIR resources in a Bundle.

    Parameters
    ----------
    resources : list[dict]
        Already-built FHIR resource dicts.
    bundle_type : str
        FHIR bundle type – ``"collection"`` (default), ``"transaction"``,
        ``"batch"``, etc.

    Returns
    -------
    dict  – a valid FHIR R4 Bundle resource.
    """
    entries: list[dict] = []
    for res in resources:
        entry: dict[str, Any] = {
            "fullUrl": f"urn:uuid:{res.get('id', make_id())}",
            "resource": res,
        }
        # For transaction / batch bundles, add a request element
        if bundle_type in ("transaction", "batch"):
            rtype = res.get("resourceType", "Resource")
            entry["request"] = {
                "method": "PUT",
                "url": f"{rtype}/{res.get('id', '')}",
            }
        entries.append(entry)

    return {
        "resourceType": "Bundle",
        "id": make_id(),
        "type": bundle_type,
        "total": len(entries),
        "entry": entries,
    }
