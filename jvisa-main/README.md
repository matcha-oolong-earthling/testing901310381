# JVISA
## Project Goal
Team J-VISA proposes SepsisAlert, an AI-powered, SMART on FHIR-enabled sepsis early warning application.

## Project Plan
A google docs is available [here](https://docs.google.com/document/d/1YR0c7FaWq_x-acDKTmlTiUw1TnhEcQDkG0ikE1Laq94/edit?tab=t.0#heading=h.fxylkjlls42)

## Team
```
Jeonghun Park (jpark3364)
Vatsal H Kevadiya (vkevadiya3)
Isha Das (idas35)
Siyoung Kim (skim3609)
Aditi Verma (averma466)
```

## Installation
```
pip install -e .
```

## Getting FHIR-compatible Data
```
cd scripts/
python run_fhir_mapper.py
```
`bundles.json` and `bundles.ndjson` are located at `dataset/MIMIC-IV-ICU-synthetic`.
