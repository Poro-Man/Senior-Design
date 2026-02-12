import os
from fastapi import FastAPI
from pydantic import BaseModel

from .llm_infer import LLMInfer

CKPT = os.environ.get("LLM_CKPT", "./checkpoints_local_quick/checkpoint_step_100.pt")
DEVICE = os.environ.get("LLM_DEVICE", "cpu")  # on Mac you can later try "mps" if your model supports it cleanly

app = FastAPI(title="Mini LLM Server")

llm = LLMInfer(CKPT, device=DEVICE)


class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/generate")
def generate(req: GenRequest):
    text = llm.generate(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return {"text": text}
