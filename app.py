from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Literal
from environment import SafeDigEnvironment
from models import SafeDigAction, SafeDigObservation, SafeDigState

app = FastAPI(
    title="SafeDig RL Environment",
    description="A safety-critical RL environment for mining task safety decisions",
    version="1.0.0"
)

env = SafeDigEnvironment()

# Request model with proper documentation
class ResetRequest(BaseModel):
    """Request to reset the environment"""
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    
    class Config:
        json_schema_extra = {
            "example": {
                "difficulty": "easy"
            }
        }

# ✅ Root endpoint with HTML interface
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>SafeDig RL Environment</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 40px 20px;
                }
                .container {
                    max-width: 900px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    padding: 40px;
                }
                h1 { 
                    color: #333;
                    margin-bottom: 10px;
                    font-size: 2.5em;
                }
                .subtitle { 
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 1.1em;
                }
                .section {
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }
                .section h2 {
                    color: #667eea;
                    margin-bottom: 15px;
                    font-size: 1.4em;
                }
                .endpoints {
                    list-style: none;
                }
                .endpoints li {
                    padding: 10px 0;
                    border-bottom: 1px solid #ddd;
                }
                .endpoints li:last-child {
                    border-bottom: none;
                }
                .endpoint-name {
                    background: #667eea;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-family: monospace;
                    font-weight: bold;
                }
                .actions {
                    display: flex;
                    gap: 15px;
                    margin-top: 30px;
                    flex-wrap: wrap;
                }
                .btn {
                    padding: 12px 24px;
                    border: none;
                    border-radius: 6px;
                    font-size: 1em;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    transition: transform 0.2s;
                }
                .btn:hover {
                    transform: translateY(-2px);
                }
                .btn-primary {
                    background: #667eea;
                    color: white;
                }
                .btn-secondary {
                    background: #6c757d;
                    color: white;
                }
                .btn-success {
                    background: #28a745;
                    color: white;
                }
                code {
                    background: #eee;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: monospace;
                }
                .tasks {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }
                .task-card {
                    background: white;
                    border: 2px solid #667eea;
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                }
                .task-card h3 {
                    color: #667eea;
                    margin-bottom: 8px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⛏️ SafeDig RL Environment</h1>
                <p class="subtitle">AI Agent Safety Decision Making for Mining Operations</p>
                
                <div class="section">
                    <h2>🎯 Problem</h2>
                    <p>An AI agent must decide whether to approve, postpone, scale down, or mandate safety measures for hazardous mining tasks based on real-time sensor data.</p>
                </div>
                
                <div class="section">
                    <h2>📊 Task Difficulties</h2>
                    <div class="tasks">
                        <div class="task-card">
                            <h3>🟢 Easy</h3>
                            <p>Clearly safe or clearly dangerous scenarios</p>
                        </div>
                        <div class="task-card">
                            <h3>🟡 Medium</h3>
                            <p>Mixed conditions requiring careful analysis</p>
                        </div>
                        <div class="task-card">
                            <h3>🔴 Hard</h3>
                            <p>Borderline sensor values - nuanced judgment</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔗 API Endpoints</h2>
                    <ul class="endpoints">
                        <li><span class="endpoint-name">POST /reset</span> - Start a new episode with difficulty level</li>
                        <li><span class="endpoint-name">POST /step</span> - Take an action (decision + reasoning)</li>
                        <li><span class="endpoint-name">GET /state</span> - Get current environment state</li>
                        <li><span class="endpoint-name">GET /health</span> - Health check endpoint</li>
                        <li><span class="endpoint-name">GET /docs</span> - Interactive Swagger UI (Try API here!)</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🎮 Actions Available</h2>
                    <ul class="endpoints">
                        <li><code>approve</code> - Approve the task (high reward if safe, severe penalty if danger)</li>
                        <li><code>postpone</code> - Delay task (high reward if danger, small penalty if safe)</li>
                        <li><code>scale_down</code> - Reduce task scope (partial reward both ways)</li>
                        <li><code>mandate_safety</code> - Add extra safety (good in danger, slight waste if safe)</li>
                    </ul>
                </div>
                
                <div class="actions">
                    <a href="/docs" class="btn btn-primary">📖 Try Interactive API</a>
                    <a href="https://github.com/spiderweb2006/openenv-project" class="btn btn-secondary">💻 View Code</a>
                </div>
            </div>
        </body>
    </html>
    """

# Reset endpoint
@app.post("/reset", response_model=dict)
def reset(req: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.
    
    - **difficulty**: Choose from 'easy', 'medium', or 'hard'
    """
    obs = env.reset(difficulty=req.difficulty)
    return obs.model_dump()

# Step endpoint
@app.post("/step", response_model=dict)
def step(action: SafeDigAction):
    """
    Take an action in the environment.
    
    - **decision**: One of 'approve', 'postpone', 'scale_down', 'mandate_safety'
    - **reasoning**: Explanation for the decision (optional)
    """
    obs = env.step(action)
    return obs.model_dump()

# State endpoint
@app.get("/state", response_model=dict)
def state():
    """Get the current state of the environment."""
    return env.state.model_dump()

# Health check
@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "ready"}

# Exception handler
@app.get("/")
def index():
    """Root endpoint - redirects to HTML interface."""
    return {"message": "SafeDig RL Environment is running. Visit the web interface above."}