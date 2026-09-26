import os
import json
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from google import genai
from google.genai import types
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

KNOWLEDGE_PATH = os.path.join(
    BASE_DIR,
    "tea_clone_knowledge.json"
)


# Load model
model = tf.keras.models.load_model(MODEL_PATH)


# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


# Load Tea Clone knowledge
with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    tea_knowledge = json.load(f)


# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# -----------------------------
# Gemini AI Chat Configuration
# -----------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# Load tea clone knowledge
KNOWLEDGE_PATH = os.path.join(
    BASE_DIR,
    "tea_clone_knowledge.json"
)

with open(KNOWLEDGE_PATH, "r") as f:
    tea_knowledge = json.load(f)


app = FastAPI(
    title="Tea Clone Classification API",
    description="Deep Learning API for Low-Country Tea Clone Identification",
    version="1.1.0"
)


@app.get("/")
def home():
    return {
        "message": "Tea Clone Classification API is running",
        "model": "MobileNetV3Large",
        "classes": class_names,
        "chat_available": gemini_client is not None
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "chat_available": gemini_client is not None
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


# -------------------------
# Tea Clone Chatbot
# -------------------------

class ChatRequest(BaseModel):
    question: str
    clone: str | None = None


@app.post("/chat")
async def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    if gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini API is not configured. Please set GEMINI_API_KEY."
        )

    # Convert the knowledge base to readable JSON text.
    knowledge_text = json.dumps(
        tea_knowledge,
        indent=2,
        ensure_ascii=False
    )

    clone_context = request.clone if request.clone else "Not specified"

    system_instruction = """
You are the Tea Clone Assistant for a tea clone identification mobile application.

Your primary knowledge source is the provided T.R.I. Advisory Circular knowledge base.

IMPORTANT RULES:
1. Answer questions using the provided knowledge base as the primary source.
2. Do not invent tea clone characteristics, ratings, regions, or recommendations.
3. If the knowledge base does not contain enough information, clearly say that the provided circular does not contain enough information.
4. Do not present outside knowledge as if it came from the circular.
5. If a characteristic is marked "Not specifically rated", say that it was not specifically rated. Do not interpret it as low, poor, or unsuitable.
6. The source circular was issued in December 2002. Mention this when the date/context is relevant.
7. Give concise, easy-to-understand answers suitable for a mobile application.
8. If the user asks something unrelated to tea clones or the provided circular, politely say that you are designed to answer questions about the tea-clone information.
"""

    user_prompt = f"""
Current identified clone: {clone_context}

Tea Clone Knowledge Base:
{knowledge_text}

User question:
{request.question}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=500
            )
        )

        answer = response.text

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        return {
            "success": True,
            "answer": answer,
            "clone": request.clone
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat service error: {str(e)}"
        )


# -----------------------------
# AI Chat
# -----------------------------

class ChatRequest(BaseModel):
    question: str
    clone: str | None = None


@app.post("/chat")
async def chat(request: ChatRequest):

    if not gemini_client:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured."
        )

    clone_info = ""

    if request.clone:
        clone_data = tea_knowledge.get(
            "target_clones",
            {}
        ).get(request.clone)

        if clone_data:
            clone_info = json.dumps(
                {
                    request.clone: clone_data
                },
                indent=2
            )

    prompt = f"""
You are a helpful assistant for a Tea Clone Identification application.

Your main knowledge source is the provided T.R.I. Advisory Circular
information below.

IMPORTANT RULES:
1. Answer using the provided knowledge.
2. Do not invent tea clone characteristics.
3. If the information is not available, clearly say that the provided
   T.R.I. circular does not contain enough information.
4. "Not specifically rated" does NOT mean low, poor, or unsuitable.
5. The circular was issued in December 2002.
6. Keep answers simple and easy for a tea grower or student to understand.

User question:
{request.question}

Identified clone:
{request.clone if request.clone else "Not provided"}

Information about the identified clone:
{clone_info}

General T.R.I. information:
{json.dumps(tea_knowledge.get("general_guidance", []), indent=2)}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return {
            "success": True,
            "answer": response.text,
            "clone": request.clone
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
