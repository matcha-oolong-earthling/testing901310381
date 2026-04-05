"""
jvisa.csv_mapper – Convert FHIR R4 Bundles back into a flat pandas DataFrame.

Quick start
-----------
>>> from jvisa.csv_mapper import FHIRToDataFrameMapper
>>> mapper = FHIRToDataFrameMapper()
>>> df = mapper.from_ndjson("dataset/MIMIC-IV-ICU-synthetic/bundles.ndjson")
>>> df = mapper.from_json("dataset/MIMIC-IV-ICU-synthetic/bundles.json")
"""

from .parser import FHIRToDataFrameMapper

__all__ = ["FHIRToDataFrameMapper"]
__version__ = "0.1.0"
