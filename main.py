from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("ridge_poly3_model.pk1")

# Initialize API
app = FastAPI(title="Estate Price Prediction", version="1.0")

# Define Request Schema with named fields
class InputData(BaseModel):
    grade: int
    sqft_living: float
    bathrooms: float
    view: int
    yr_renovated: int
    bedrooms: int
    zipcode: int
    lat: float
    long: float

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame in correct column order
    features = pd.DataFrame([[
        data.grade, data.sqft_living, data.bathrooms, data.view,
        data.yr_renovated, data.bedrooms, data.zipcode, data.lat, data.long
    ]], columns=[
        "grade", "sqft_living", "bathrooms", "view",
        "yr_renovated", "bedrooms", "zipcode", "lat", "long"
    ])

    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}


