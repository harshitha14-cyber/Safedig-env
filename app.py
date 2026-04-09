from fastapi import FastAPI
from pydantic import BaseModel
from environment import SafeDigEnvironment
from models import SafeDigAction, SafeDigObservation

app = FastAPI(title="SafeDig RL Environment")
env = SafeDigEnvironment()

class ResetRequest(BaseModel):
    difficulty: str = "easy"  # easy / medium / hard

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs = env.reset(difficulty=req.difficulty)
    return obs.model_dump()

@app.post("/step")
def step(action: SafeDigAction):
    obs = env.step(action)
    return obs.model_dump()

@app.get("/state")
def state():
    return env.state.model_dump()

@app.get("/health")
def health():
    return {"status": "ok"}