import asyncio
import uvicorn
from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException, status, Path
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Load model outside the route to ensure it's loaded once
model = joblib.load("../linear_regression/best_model.joblib")

# Create an app instance 
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST requests
    allow_headers=["*"],  # Allow any headers
)

# Define the Pydantic model for the request
class WineQRequest(BaseModel):
    fixed_acidity: float = Field(gt=0, lt=1000.0)
    volatile_acidity: float = Field(gt=0, lt=1000.0)
    residual_sugar: float = Field(gt=0, lt=1000.0)
    chlorides: float = Field(gt=0, lt=1000.0)
    free_SO2: float = Field(gt=0, lt=1000.0)
    sulphates: float = Field(gt=0, lt=1000.0)
    alcohol: float = Field(gt=0, lt=1000.0)
    colour: int = Field(gt=-1, lt=2)  # Assuming colour is a binary value (0 or 1)

# Testing endpoint to verify the API is working
@app.get("/class")
async def get_greet():
    return {"Message": "API is successfully running!"}

# Root endpoint to test if the server is up
@app.get("/", status_code=status.HTTP_200_OK)
async def get_hello():
    return {"message": "Welcome to the Wine Quality Prediction API!"}

# Endpoint to make predictions
@app.post('/predict', status_code=status.HTTP_200_OK)
async def make_prediction(wineq_request: WineQRequest):
    try:
        # Prepare the input data for the model
        single_row = [[
            wineq_request.fixed_acidity, 
            wineq_request.volatile_acidity, 
            wineq_request.residual_sugar, 
            wineq_request.chlorides, 
            wineq_request.free_SO2, 
            wineq_request.sulphates, 
            wineq_request.alcohol, 
            wineq_request.colour
        ]]
        
        # Predict the output using the loaded model
        new_value = model.predict(single_row)
        
        # Convert the predicted value to an integer (if needed)
        integer_quality = int(new_value[0])  # assuming the output should be an integer
        
        # Return the prediction
        return {"predicted Quality": integer_quality}
    
    except Exception as e:
        # Return a detailed error message in case of failure
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

 # To run the app (run this in the terminal)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
