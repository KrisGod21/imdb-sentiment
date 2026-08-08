"""
Sentiment classifier API.

Loads the fine-tuned DistilBERT model (from ./sentiment_model, or from the
Hugging Face Hub if MODEL_SOURCE is set to a hub repo id) and exposes a
/predict endpoint. Also serves the static frontend at /.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

# If you pushed your model to the Hugging Face Hub, set this env var to
# "your-username/sentiment-distilbert" instead of using a local folder.
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "./sentiment_model")

app = FastAPI(title="Sentiment Classifier API")

# Allow the frontend (even if hosted elsewhere) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None


@app.on_event("startup")
def load_model():
    global classifier
    classifier = pipeline("text-classification", model=MODEL_SOURCE, tokenizer=MODEL_SOURCE)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty")
    if len(req.text) > 5000:
        raise HTTPException(status_code=400, detail="text too long (max 5000 chars)")

    result = classifier(req.text[:2000])[0]
    return PredictResponse(label=result["label"], confidence=round(result["score"], 4))


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the frontend (index.html + assets) at the root URL
app.mount("/", StaticFiles(directory="static", html=True), name="static")
