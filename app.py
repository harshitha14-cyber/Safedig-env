from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from environment import SafeDigEnvironment
from models import SafeDigAction, SafeDigObservation

app = FastAPI(title="SafeDig RL Environment")
env = SafeDigEnvironment()

class ResetRequest(BaseModel):
    difficulty: str = "easy"

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>SafeDig RL Environment</title>
            <style>
                body { font-family: Arial; margin: 40px; background: #f5f5f5; }
                h1 { color: #333; }
                .info { background: white; padding: 20px; border-radius: 8px; }
                code { background: #eee; padding: 2px 6px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>⛏️ SafeDig RL Environment</h1>
            <div class="info">
                <h2>API Endpoints</h2>
                <ul>
                    <li><code>POST /reset</code> - Start a new episode</li>
                    <li><code>POST /step</code> - Take an action</li>
                    <li><code>GET /state</code> - Get current state</li>
                    <li><code>GET /health</code> - Health check</li>
                    <li><code>GET /docs</code> - Swagger UI (Interactive API)</li>
                </ul>
                <p><strong>👉 <a href="/docs">View Interactive API Docs →</a></strong></p>
            </div>
        </body>
    </html>
    """

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