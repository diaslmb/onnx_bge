from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from typing import Union
import uvicorn

MODEL_ID = "bge-m3-onnx"
MODEL_PATH = "onnx_bge/model.onnx"

tokenizer = AutoTokenizer.from_pretrained("onnx_bge")
session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

app = FastAPI()

class EmbedRequest(BaseModel):
    input: Union[str, list]
    model: str = MODEL_ID

@app.post("/v1/embeddings")
def embed(req: EmbedRequest):
    if req.model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    # Make sure we handle both single strings and lists
    inputs = [req.input] if isinstance(req.input, str) else req.input

    tokens = tokenizer(inputs, return_tensors="np", padding=True, truncation=True)
    outputs = session.run(None, {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    })
    embeddings = outputs[0][:, 0, :]
    return {"data": [{"embedding": emb.tolist(), "index": i} for i, emb in enumerate(embeddings)]}

@app.get("/v1/models")
def get_model_info():
    return {
        "id": MODEL_ID,
        "object": "model",
        "owned_by": "you",
        "ready": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
