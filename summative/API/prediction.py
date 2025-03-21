import asyncio
import uvicorn
from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException, status, Path
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import joblib

model = joblib.load("summative/linear_regression/best_model.joblib")
# creating an app instance 
app = FastAPI() 

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST requests
    allow_headers=["*"],  # Allow any headers
)

# create a pedantic class for the request
class WineQRequest(BaseModel):
    fixed_acidity: float = Field(gt=0, lt=1000.0)
    volatile_acidity: float = Field(gt=0, lt=1000.000)
    residual_sugar: float = Field(gt=0, lt=1000.0)
    chlorides: float = Field(gt=0, lt=1000.000)
    free_SO2: float = Field(gt=0, lt=1000.0)
    sulphates: float = Field(gt=0, lt=1000.000)
    alcohol: float = Field(gt=0, lt=1000.0)
    colour: int = Field(gt=-1, lt=2)


# class testing 
@app.get("/class")
async def get_greet():
    return {"Message": "Hello API"}


@app.get("/", status_code=status.HTTP_200_OK)
async def get_hello():
    return {"hello": "summative yeey"}

@app.post('/predict', status_code=status.HTTP_200_OK)
async def make_prediction(wineq_request: WineQRequest):
    try:
        single_row = [[wineq_request.fixed_acidity, wineq_request.volatile_acidity, wineq_request.residual_sugar, wineq_request.chlorides, wineq_request.free_SO2, wineq_request.sulphates, wineq_request.alcohol, wineq_request.colour]]
        new_value = model.predict(single_row)
        integer_quality = int(new_value[0])  # Convert the predicted value to an integer
        return {"predicted Quality": integer_quality}
        
    #except:
    #    raise HTTPException(status_code=500, detail="Something went wrong.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))