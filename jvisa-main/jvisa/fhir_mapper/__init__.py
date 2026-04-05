"""
jvisa.fhir_mapper – Convert MIMIC-IV ICU tabular data into FHIR R4 resources.

Quick start
-----------
>>> from jvisa.fhir_mapper import MIMICToFHIRMapper
>>> mapper = MIMICToFHIRMapper()
>>> bundle = mapper.map_row(row_dict)
>>> bundles = mapper.map_csv("dataset/MIMIC-IV-ICU-synthetic/data.csv")
"""

from .mapper import MIMICToFHIRMapper

__all__ = ["MIMICToFHIRMapper"]
__version__ = "0.1.0"
