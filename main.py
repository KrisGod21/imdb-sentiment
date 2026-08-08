"""
Sentiment classifier API — ONNX runtime version (no torch, tiny memory footprint).
"""

import os
import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODEL_REPO = os.environ.get("MODEL_SOURCE", "KrisGod/sentiment-distilbert-onnx")

app = FastAPI(title="Sentiment Classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

session = None
tokenizer = None
id2label = None


@app.on_event("startup")
def load_model():
    global session, tokenizer, id2label

    model_path = hf_hub_download(MODEL_REPO, "model.onnx")
    tokenizer_path = hf_hub_download(MODEL_REPO, "tokenizer.json")
    config_path = hf_hub_download(MODEL_REPO, "config.json")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=256)

    with open(config_path) as f:
        config = json.load(f)
    id2label = {int(k): v for k, v in config["id2label"].items()}


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


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

    encoding = tokenizer.encode(req.text[:2000])
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

    input_names = [i.name for i in session.get_inputs()]
    feed = {}
    if "input_ids" in input_names:
        feed["input_ids"] = input_ids
    if "attention_mask" in input_names:
        feed["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(input_ids)

    logits = session.run(None, feed)[0][0]
    probs = softmax(logits)
    idx = int(np.argmax(probs))

    return PredictResponse(label=id2label[idx], confidence=round(float(probs[idx]), 4))


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")