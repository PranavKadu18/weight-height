from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI()


# Define request model
class HeightData(BaseModel):
    height: float


# Load the trained model
try:
    model = joblib.load('weight_prediction_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Please ensure you've run train_model.py first to create the model file.")
    raise

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve the HTML file
@app.get("/")
async def read_root():
    return FileResponse("index.html")


# Prediction endpoint
@app.post("/predict")
async def predict(data: HeightData):
    try:
        # Input validation
        if data.height <= 0:
            raise HTTPException(status_code=400, detail="Height must be positive")

        # Make prediction
        height_reshaped = np.array([[data.height]])
        prediction = model.predict(height_reshaped)

        return {"weight": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
