"""
Standard medical code system mappings (LOINC, SNOMED CT, ICD-10-CM)
for translating MIMIC-IV ICU data fields into FHIR-compatible codings.
"""

# ---------------------------------------------------------------------------
# System URIs
# ---------------------------------------------------------------------------
LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
FHIR_OBS_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
FHIR_CONDITION_CATEGORY = (
    "http://terminology.hl7.org/CodeSystem/condition-category"
)
UCUM = "http://unitsofmeasure.org"
V3_ETHNICITY = "urn:oid:2.16.840.1.113883.6.238"
V3_ADMIN_GENDER = "http://hl7.org/fhir/administrative-gender"
ENCOUNTER_CLASS_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-ActCode"

# ---------------------------------------------------------------------------
# Vital-sign observations  (column_name -> LOINC mapping)
# Each value: (loinc_code, display, unit, ucum_unit)
# ---------------------------------------------------------------------------
VITAL_SIGNS: dict[str, tuple[str, str, str, str]] = {
    # Heart rate
    "hr_mean":  ("8867-4", "Heart rate (mean)", "/min", "/min"),
    "hr_max":   ("8867-4", "Heart rate (max)", "/min", "/min"),
    "hr_min":   ("8867-4", "Heart rate (min)", "/min", "/min"),
    # Blood pressure – systolic
    "sbp_mean": ("8480-6", "Systolic BP (mean)", "mmHg", "mm[Hg]"),
    "sbp_max":  ("8480-6", "Systolic BP (max)", "mmHg", "mm[Hg]"),
    "sbp_min":  ("8480-6", "Systolic BP (min)", "mmHg", "mm[Hg]"),
    # Blood pressure – diastolic
    "dbp_mean": ("8462-4", "Diastolic BP (mean)", "mmHg", "mm[Hg]"),
    "dbp_max":  ("8462-4", "Diastolic BP (max)", "mmHg", "mm[Hg]"),
    "dbp_min":  ("8462-4", "Diastolic BP (min)", "mmHg", "mm[Hg]"),
    # Mean arterial pressure
    "map_mean": ("8478-0", "Mean arterial pressure", "mmHg", "mm[Hg]"),
    # Temperature
    "temp_celsius_mean": ("8310-5", "Body temperature (mean)", "°C", "Cel"),
    "temp_celsius_max":  ("8310-5", "Body temperature (max)", "°C", "Cel"),
    "temp_celsius_min":  ("8310-5", "Body temperature (min)", "°C", "Cel"),
    # SpO2
    "spo2_mean": ("2708-6", "SpO2 (mean)", "%", "%"),
    "spo2_min":  ("2708-6", "SpO2 (min)", "%", "%"),
    "spo2_max":  ("2708-6", "SpO2 (max)", "%", "%"),
    # Respiratory rate
    "respiratory_rate_mean": ("9279-1", "Respiratory rate (mean)", "/min", "/min"),
    "respiratory_rate_max":  ("9279-1", "Respiratory rate (max)", "/min", "/min"),
    "respiratory_rate_min":  ("9279-1", "Respiratory rate (min)", "/min", "/min"),
}

# ---------------------------------------------------------------------------
# Laboratory observations  (column_name -> LOINC mapping)
# Each value: (loinc_code, display, unit, ucum_unit)
# ---------------------------------------------------------------------------
LAB_RESULTS: dict[str, tuple[str, str, str, str]] = {
    "wbc":              ("6690-2",  "WBC",                  "10^3/uL", "10*3/uL"),
    "lactate_mmol":     ("2524-7",  "Lactate",              "mmol/L",  "mmol/L"),
    "creatinine":       ("2160-0",  "Creatinine",           "mg/dL",   "mg/dL"),
    "platelet_count":   ("777-3",   "Platelet count",       "10^3/uL", "10*3/uL"),
    "bilirubin_total":  ("1975-2",  "Bilirubin, total",     "mg/dL",   "mg/dL"),
    "glucose":          ("2345-7",  "Glucose",              "mg/dL",   "mg/dL"),
    "ph_arterial":      ("2744-1",  "Arterial pH",          "",        "[pH]"),
    "pao2_fio2_ratio":  ("50984-4", "PaO2/FiO2 ratio",     "mmHg",    "mm[Hg]"),
    "inr":              ("6301-6",  "INR",                  "",        "{INR}"),
    "sodium":           ("2951-2",  "Sodium",               "mEq/L",   "meq/L"),
    "potassium":        ("2823-3",  "Potassium",            "mEq/L",   "meq/L"),
    "chloride":         ("2075-0",  "Chloride",             "mEq/L",   "meq/L"),
    "bicarbonate":      ("1963-8",  "Bicarbonate",          "mEq/L",   "meq/L"),
    "hematocrit":       ("4544-3",  "Hematocrit",           "%",       "%"),
    "hemoglobin":       ("718-7",   "Hemoglobin",           "g/dL",    "g/dL"),
}

# ---------------------------------------------------------------------------
# Clinical scoring observations
# ---------------------------------------------------------------------------
SCORES: dict[str, tuple[str, str]] = {
    "sofa_score":    ("96790-1", "SOFA score"),
    "apache_iv":     ("75889-6", "APACHE IV score"),
    "qsofa":         ("96792-7", "qSOFA score"),
    "sirs_criteria": ("35925-4", "SIRS criteria count"),
    "gcs_total":     ("9269-2",  "Glasgow Coma Scale total"),
    "fio2_percent":  ("3150-0",  "FiO2"),
    "sedation_score": ("54530-1", "Sedation score"),
}

# ---------------------------------------------------------------------------
# Comorbidity / condition flags  (column_name -> ICD-10 + SNOMED)
# Each value: (icd10_code, snomed_code, display)
# ---------------------------------------------------------------------------
CONDITIONS: dict[str, tuple[str, str, str]] = {
    "diabetes":               ("E11",    "73211009",  "Diabetes mellitus"),
    "hypertension":           ("I10",    "38341003",  "Hypertension"),
    "chf":                    ("I50",    "42343007",  "Congestive heart failure"),
    "copd":                   ("J44",    "13645005",  "COPD"),
    "chronic_kidney_disease": ("N18",    "709044004", "Chronic kidney disease"),
    "liver_disease":          ("K76",    "235856003", "Liver disease"),
    "immunosuppression":      ("D84.9",  "234532001", "Immunosuppression"),
    "cad":                    ("I25.1",  "53741008",  "Coronary artery disease"),
    "atrial_fibrillation":    ("I48",    "49436004",  "Atrial fibrillation"),
    "cancer_active":          ("C80",    "363346000", "Active cancer"),
    "sepsis_label":           ("A41.9",  "91302008",  "Sepsis"),
}

# ---------------------------------------------------------------------------
# ICU procedures / interventions  (column_name -> SNOMED)
# Each value: (snomed_code, display)
# ---------------------------------------------------------------------------
PROCEDURES: dict[str, tuple[str, str]] = {
    "vasopressors_flag":       ("398162007", "Vasopressor therapy"),
    "mechanical_ventilation":  ("40617009",  "Mechanical ventilation"),
    "antibiotics_24h":         ("281789004", "Antibiotic therapy"),
    "insulin_infusion_flag":   ("14602007",  "Insulin infusion"),
}

# ---------------------------------------------------------------------------
# Body-measurement observations (weight, height, BMI)
# ---------------------------------------------------------------------------
BODY_MEASUREMENTS: dict[str, tuple[str, str, str, str]] = {
    "weight_kg":  ("29463-7", "Body weight",     "kg", "kg"),
    "height_cm":  ("8302-2",  "Body height",     "cm", "cm"),
    "bmi":        ("39156-5", "Body mass index",  "kg/m2", "kg/m2"),
}

# ---------------------------------------------------------------------------
# Gender mapping
# ---------------------------------------------------------------------------
GENDER_MAP: dict[str, str] = {
    "M": "male",
    "F": "female",
    "O": "other",
    "U": "unknown",
}

# ---------------------------------------------------------------------------
# Ethnicity → HL7 race/ethnicity codes (simplified)
# ---------------------------------------------------------------------------
ETHNICITY_MAP: dict[str, tuple[str, str]] = {
    "White":    ("2106-3", "White"),
    "Black":    ("2054-5", "Black or African American"),
    "Asian":    ("2028-9", "Asian"),
    "Hispanic": ("2135-2", "Hispanic or Latino"),
    "Other":    ("2131-1", "Other Race"),
}
