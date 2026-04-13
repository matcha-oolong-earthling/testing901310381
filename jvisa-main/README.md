# JVISA
## Project Goal
Team J-VISA proposes SepsisAlert a SMART on FHIR-enabled sepsis early warning application.

## Project Plan
A google docs is available [here](https://docs.google.com/document/d/1YR0c7FaWq_x-acDKTmlTiUw1TnhEcQDkG0ikE1Laq94/edit?tab=t.0#heading=h.fxylkjlls42)

## Team
```
Jeonghun Park (jpark3364)
Vatsal H Kevadiya (vkevadiya3)
Isha Das (idas35) personal github user name (matcha-oolong-earthling)
Siyoung Kim (skim3609)
Aditi Verma (averma466)
```

## Installation
```bash
pip install -e .
```

## Project Structure
```
jvisa/
  fhir_mapper/    # CSV -> FHIR R4 Bundles
  csv_mapper/     # FHIR R4 Bundles -> pandas DataFrame
  model/          # Random Forest sepsis classifier
scripts/
  run_fhir_mapper.py      # CSV -> FHIR conversion
  run_sepsis_model.py     # FHIR -> DataFrame -> model training
  plot_model_results.py   # Generate evaluation plots
app.py                    # Streamlit web application
```

## Getting FHIR-compatible Data
```bash
cd scripts/
python run_fhir_mapper.py
```
`bundles.json` and `bundles.ndjson` are located at `dataset/MIMIC-IV-ICU-synthetic`.

## Running the Sepsis Model
Convert FHIR bundles back to a DataFrame and train the Random Forest classifier:
```bash
python scripts/run_sepsis_model.py
```

Options:
```bash
python scripts/run_sepsis_model.py --noise-scale 2.0    # add noise for realistic evaluation
python scripts/run_sepsis_model.py --impute mean         # change imputation strategy (median, mean, zero)
python scripts/run_sepsis_model.py --n-estimators 300    # number of trees
python scripts/run_sepsis_model.py --max-depth 10        # limit tree depth
```

## Generating Plots
```bash
python scripts/plot_model_results.py
```
Output saved to `figures/sepsis_rf_results.png`.

## Streamlit Web App
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with four tabs:
- **Overview** -- pipeline architecture, dataset summary, feature categories
- **Data Explorer** -- browse parsed DataFrame, summary statistics, distribution plots
- **Model Results** -- train the Random Forest with configurable hyperparameters, view confusion matrix, feature importances, and ROC curve
- **Patient Lookup** -- select individual patients and view sepsis risk predictions
