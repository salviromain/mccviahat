import os
import time

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# This lets you change the model without changing code:
# e.g. MODEL_ID=distilgpt2
MODEL_ID = os.environ.get("MODEL_ID", "gpt2")

app = FastAPI(title="mccviahat-llm")

class GenerateRequest(BaseModel):
    text: str
    max_tokens: int = 50

# Load model once when the server starts (not per-request)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()  # inference mode

@app.get("/health")
def health():
    # quick endpoint to verify the server is running
    return {"status": "ok", "model": MODEL_ID}

@app.post("/generate")
def generate(req: GenerateRequest):
    t0 = time.time()

    inputs = tokenizer(req.text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=req.max_tokens,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    latency_ms = int((time.time() - t0) * 1000)

    return {"response": text, "latency_ms": latency_ms, "model": MODEL_ID}
