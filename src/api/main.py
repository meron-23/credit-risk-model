from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

from src.api.pydantic_models import RiskRequest, RiskResponse

app = FastAPI()

# Load model from MLflow
model = mlflow.pyfunc.load_model("models:/best_model/Production")

@app.get("/")
def home():
    return {"message": "Risk Prediction API is running"}

@app.post("/predict", response_model=RiskResponse)
def predict_risk(data: RiskRequest):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"risk_probability": prediction[0]}
