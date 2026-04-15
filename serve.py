# serve.py — Credit Risk Prediction API
# Run with: uvicorn serve:app --reload
# Docs at:  http://localhost:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ── Load artefacts ────────────────────────────────────────────────────────────
BASE = Path("model_artifacts")

try:
    model     = joblib.load(BASE / "credit_risk_model.pkl")
    threshold = joblib.load(BASE / "optimal_threshold.pkl")
    with open(BASE / "model_schema.json") as f:
        schema = json.load(f)
    print(f"✅ Model loaded: {schema['model_name']}  |  AUC={schema['auc_test']}  |  KS={schema['ks_test']}")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artefact not found: {e}\n"
        "Run the training notebook first to generate model_artifacts/"
    )

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts loan default probability, credit score, and approve/decline decision.",
    version="1.0.0",
)

# ── Request schema ────────────────────────────────────────────────────────────
class ApplicantIn(BaseModel):
    person_age:                 float = Field(..., ge=18,  le=100,  example=32,     description="Applicant age in years")
    person_income:              float = Field(..., gt=0,            example=75000,  description="Annual income in USD")
    person_emp_length:          float = Field(None, ge=0,           example=5.0,    description="Employment length in years (null if unknown)")
    loan_amnt:                  float = Field(..., gt=0,            example=12000,  description="Requested loan amount in USD")
    loan_int_rate:              float = Field(None, ge=1,  le=60,   example=11.5,   description="Interest rate % (null if unknown)")
    loan_percent_income:        float = Field(..., ge=0,  le=1,     example=0.16,   description="Loan amount as fraction of income")
    cb_person_cred_hist_length: float = Field(..., ge=0,            example=8.0,    description="Credit history length in years")
    loan_grade:                 str   = Field(...,                  example="B",    description="Loan grade assigned by lender (A–G)")
    person_home_ownership:      str   = Field(...,                  example="RENT", description="RENT | OWN | MORTGAGE | OTHER")
    loan_intent:                str   = Field(...,                  example="PERSONAL", description="Purpose of the loan")
    cb_person_default_on_file:  str   = Field(...,                  example="N",    description="Prior default on file: Y or N")

    @validator("loan_grade")
    def validate_grade(cls, v):
        v = v.upper()
        if v not in list("ABCDEFG"):
            raise ValueError("loan_grade must be one of A, B, C, D, E, F, G")
        return v

    @validator("person_home_ownership")
    def validate_ownership(cls, v):
        v = v.upper()
        if v not in {"RENT", "OWN", "MORTGAGE", "OTHER"}:
            raise ValueError("person_home_ownership must be RENT, OWN, MORTGAGE, or OTHER")
        return v

    @validator("cb_person_default_on_file")
    def validate_default_flag(cls, v):
        v = v.upper()
        if v not in {"Y", "N"}:
            raise ValueError("cb_person_default_on_file must be Y or N")
        return v

    @validator("loan_intent")
    def validate_intent(cls, v):
        allowed = {"PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"}
        if v.upper() not in allowed:
            raise ValueError(f"loan_intent must be one of {allowed}")
        return v.upper()

# ── Response schema ───────────────────────────────────────────────────────────
class PredictionOut(BaseModel):
    default_probability: float = Field(..., description="Probability of default (0–1)")
    credit_score:        int   = Field(..., description="Derived credit score (300–850)")
    decision:            str   = Field(..., description="APPROVE or DECLINE")
    risk_tier:           str   = Field(..., description="Very Low | Low | Medium | High | Very High")

class BatchOut(BaseModel):
    results: list[PredictionOut]
    total:   int

class HealthOut(BaseModel):
    status:     str
    model_name: str
    auc:        float
    ks:         float
    gini:       float
    threshold:  float

# ── Feature engineering (mirrors the training notebook exactly) ───────────────
def engineer_features(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])

    # Missing-value flags
    df["person_emp_length_missing"] = df["person_emp_length"].isnull().astype(int)
    df["loan_int_rate_missing"]     = df["loan_int_rate"].isnull().astype(int)

    # Engineered features
    df["log_income"]         = np.log1p(df["person_income"])
    df["dti_ratio"]          = df["loan_amnt"] / (df["person_income"] + 1)
    df["loan_per_hist_year"] = df["loan_amnt"] / (df["cb_person_cred_hist_length"] + 1)
    df["age_emp_ratio"]      = df["person_emp_length"].fillna(0) / (df["person_age"] + 1)

    grade_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
    df["grade_score"] = df["loan_grade"].map(grade_map)

    tier_map = {"A":"low","B":"low","C":"medium","D":"medium",
                "E":"high","F":"high","G":"high"}
    df["loan_risk_tier"] = df["loan_grade"].map(tier_map)

    df["prior_default"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    return df

# ── Credit score conversion ───────────────────────────────────────────────────
def prob_to_score(prob: float, base_score: int = 600,
                  base_odds: int = 50, pdo: int = 20) -> int:
    """Convert default probability → 300-850 credit score (higher = safer)."""
    prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    odds   = (1 - prob) / prob
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    raw    = int(round(offset + factor * np.log(odds)))
    return int(np.clip(raw, 300, 850))

# ── Risk tier label ───────────────────────────────────────────────────────────
def get_risk_tier(prob: float) -> str:
    if prob < 0.10:   return "Very Low"
    elif prob < 0.20: return "Low"
    elif prob < 0.35: return "Medium"
    elif prob < 0.50: return "High"
    else:             return "Very High"

# ── Core prediction logic ─────────────────────────────────────────────────────
def _predict_one(applicant_dict: dict) -> PredictionOut:
    df   = engineer_features(applicant_dict)
    prob = float(model.predict_proba(df)[0, 1])
    return PredictionOut(
        default_probability = round(prob, 4),
        credit_score        = prob_to_score(prob),
        decision            = "DECLINE" if prob >= threshold else "APPROVE",
        risk_tier           = get_risk_tier(prob),
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthOut, tags=["Health"])
def health():
    """Health check — returns model metadata and key metrics."""
    return HealthOut(
        status     = "ok",
        model_name = schema["model_name"],
        auc        = schema["auc_test"],
        ks         = schema["ks_test"],
        gini       = schema["gini_test"],
        threshold  = float(threshold),
    )

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict(applicant: ApplicantIn):
    """
    Predict default risk for a single loan applicant.

    Returns:
    - **default_probability**: 0–1 probability of default
    - **credit_score**: derived score on 300–850 scale
    - **decision**: APPROVE or DECLINE
    - **risk_tier**: Very Low / Low / Medium / High / Very High
    """
    try:
        return _predict_one(applicant.dict())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/predict/batch", response_model=BatchOut, tags=["Prediction"])
def predict_batch(applicants: list[ApplicantIn]):
    """
    Predict default risk for multiple applicants in one request.
    Returns a list of predictions in the same order as the input.
    """
    try:
        results = [_predict_one(a.dict()) for a in applicants]
        return BatchOut(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/threshold", tags=["Model Info"])
def get_threshold():
    """Returns the current decision threshold and what it means."""
    return {
        "threshold": float(threshold),
        "meaning": (
            f"Applicants with predicted default probability >= {threshold:.2f} "
            "are DECLINED. This threshold was optimised on the test set to "
            "maximise F1-score for the default (minority) class."
        )
    }
