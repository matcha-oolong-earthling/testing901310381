from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import shap
import pathlib
import numpy as np

from jvisa.csv_mapper import FHIRToDataFrameMapper
from jvisa.model import SepsisRandomForest

# Global variable to hold our trained model and SHAP explainer
model = None
explainer = None
imputer_values = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, imputer_values
    print("Initializing model...")
    
    mapper = FHIRToDataFrameMapper()
    data_path = pathlib.Path(__file__).parent.parent / "dataset" / "MIMIC-IV-ICU-synthetic" / "bundles.ndjson"
    
    if not data_path.exists():
        print(f"Warning: Data path {data_path} does not exist. The model cannot be trained on startup.")
    else:
        print("Loading training data and training model...")
        df = mapper.from_ndjson(data_path)
        
        # Calculate median values for imputation globally since a single row won't have a valid median
        numeric_cols = df.select_dtypes(include="number").columns
        imputer_values = df[numeric_cols].median()
        
        df = mapper.impute(df, strategy="median")
        
        model = SepsisRandomForest(n_estimators=200, random_state=42)
        model.train_and_evaluate(df)
        
        # Prepare background data for SHAP explainer
        X_train, _ = model._prepare(df)
        
        print("Training SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model.clf)
        print("Initialization complete.")
        
    yield
    
    model = None
    explainer = None
    imputer_values = None

app = FastAPI(title="SepsisAlert API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    bundle: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "name": "SepsisAlert API",
        "status": "online",
        "model_loaded": model is not None,
        "docs_url": "/docs"
    }

@app.post("/predict")
async def predict(payload: PredictRequest):
    if model is None or explainer is None:
        raise HTTPException(status_status_code=503, detail="Model is not initialized.")
        
    try:
        # 1. Convert incoming FHIR JSON into a pandas DataFrame
        mapper = FHIRToDataFrameMapper()
        patient_row = mapper.parse_bundle(payload.bundle)
        df_patient = pd.DataFrame([patient_row])
        
        # 2. Impute missing values using global medians from training
        for col in df_patient.select_dtypes(include="number").columns:
            if col in imputer_values:
                df_patient[col] = df_patient[col].fillna(imputer_values[col])
        
        # 3. Predict probability
        proba = model.predict_proba(df_patient)
        if len(proba[0]) == 2:
            sepsis_risk_score = proba[0][1]
        else:
            sepsis_risk_score = proba[0][0]
            
        # 4. Extract SHAP values
        X_encoded = model._encode(df_patient)
        
        # In scikit-learn random forests, TreeExplainer returns a list of arrays for classification
        # We want the SHAP values for the positive class (class 1)
        shap_values = explainer.shap_values(X_encoded)
        
        if isinstance(shap_values, list):
            shap_vals_class1 = shap_values[1][0]
        else:
            # Depending on SHAP version, it might return a single array
            if len(shap_values.shape) == 3:
                shap_vals_class1 = shap_values[0, :, 1]
            else:
                shap_vals_class1 = shap_values[0]
                
        # Combine feature names with their SHAP values
        feature_names = X_encoded.columns.tolist()
        feature_importances = dict(zip(feature_names, shap_vals_class1))
        
        # Sort by absolute SHAP value to find top 4
        top_4 = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
        
        return {
            "sepsis_risk_score": float(sepsis_risk_score),
            "top_features": {
                name: float(val) for name, val in top_4
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
