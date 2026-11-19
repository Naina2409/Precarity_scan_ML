"""FastAPI scoring service.  Run: uvicorn api.main:app --reload
Optional auth: set PRECARITY_API_KEY and send it in the `X-API-Key` header."""
import json, os
from typing import List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from src.config import MODELS_DIR
from src.data_generator import CITIES, PLATFORMS
from src.predict import load_model, predict_one

app = FastAPI(title="PrecarityScan API", version="2.0",
              description="Precarity index and financial-stress risk for gig workers. Indicator only; not financial advice.")


def check_key(x_api_key: Optional[str] = Header(default=None)):
    key = os.getenv("PRECARITY_API_KEY")
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class Worker(BaseModel):
    city: str = Field(examples=["Delhi"])
    platform: str = Field(examples=["Zomato"])
    age: int = Field(ge=16, le=80)
    months_experience: int = Field(ge=0, le=360)
    hours_per_day: float = Field(ge=1, le=18)
    days_per_week: int = Field(ge=1, le=7)
    monthly_income: float = Field(ge=0, le=300000)
    dependents: int = Field(ge=0, le=10)
    savings_months: float = Field(ge=0, le=24)
    has_health_insurance: bool
    has_paid_leave: bool
    has_retirement: bool
    multiple_platforms: bool
    has_work_loan: bool
    gender: Optional[str] = None
    monthly_incomes: Optional[List[float]] = Field(default=None, description="Last 2-12 monthly earnings (improves volatility estimate)")


@app.get("/health")
def health():
    try:
        b = load_model(); return {"status": "ok", "model_version": b["version"]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/model-info", dependencies=[Depends(check_key)])
def model_info():
    reg = MODELS_DIR / "registry.json"
    if not reg.exists(): raise HTTPException(status_code=503, detail="No model registered")
    latest = json.loads((MODELS_DIR / "latest.json").read_text())["version"]
    return next(e for e in json.loads(reg.read_text()) if e["version"] == latest)


@app.get("/options")
def options():
    return {"cities": list(CITIES), "platforms": list(PLATFORMS)}


@app.post("/predict", dependencies=[Depends(check_key)])
def predict(w: Worker):
    try:
        return predict_one(w.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
