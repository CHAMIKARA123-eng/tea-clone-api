import os
import json
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io


# Get the project directory automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "configs",
    "class_names.json"
)


# Load model
model = tf.keras.models.load_model(MODEL_PATH)


# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


app = FastAPI(
    title="Tea Clone Classification API",
    description="Deep Learning API for Low-Country Tea Clone Identification",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Tea Clone Classification API is running",
        "model": "MobileNetV3Large",
        "classes": class_names
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:

        # Read uploaded image
        image_bytes = await file.read()

        # Open image
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert to NumPy array
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = int(
            np.argmax(predictions[0])
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            predictions[0][predicted_index]
        )

        return {
            "success": True,
            "predicted_clone": predicted_class,
            "confidence": round(
                confidence * 100,
                2
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )