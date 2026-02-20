# src/server.py
import os
from fastapi import FastAPI
from pydantic import BaseModel

from src.llm_infer import LLMInfer

CKPT = os.environ.get("LLM_CKPT", "./checkpoints/pretrain_owt_1b/checkpoint_step_61036.pt")
DEVICE = os.environ.get("LLM_DEVICE", "cuda")

app = FastAPI(title="Mini LLM Server")
llm = LLMInfer(CKPT, device=DEVICE)


class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95


@app.get("/health")
def health():
    return {"status": "ok", "device": str(llm.device), "ckpt": CKPT}


@app.post("/generate")
def generate(req: GenRequest):
    text = llm.generate(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return {"text": text}
