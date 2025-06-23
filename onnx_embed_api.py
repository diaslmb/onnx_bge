from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import uvicorn

# === Model Config ===
MODEL_ID = "bge-m3-onnx"
MODEL_PATH = "onnx_bge_m3/model.onnx"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("onnx_bge_m3")
session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# === Define FastAPI app ===
app = FastAPI()

# === Request schema (OpenAI-style) ===
class EmbedRequest(BaseModel):
    input: list
    model: str = MODEL_ID  # Optional, defaults to correct model

# === Embeddings endpoint ===
@app.post("/v1/embeddings")
def embed(req: EmbedRequest):
    if req.model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    tokens = tokenizer(req.input, return_tensors="np", padding=True, truncation=True)
    outputs = session.run(None, {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    })

    # Use CLS token as embedding
    embeddings = outputs[0][:, 0, :]
    return {"data": [{"embedding": emb.tolist(), "index": i} for i, emb in enumerate(embeddings)]}

# === Model metadata endpoint ===
@app.get("/v1/models")
def get_model_info():
    return {
        "id": MODEL_ID,
        "object": "model",
        "owned_by": "you",
        "ready": True
    }

# === Launch server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
