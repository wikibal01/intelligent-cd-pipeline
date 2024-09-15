from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load('deployment_model.pkl')

@app.get("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"success": bool(prediction[0])}
