import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from environment import SafeDigEnvironment
from server.models import SafeDigAction, SafeDigObservation

env = SafeDigEnvironment()

# ========== FASTAPI APP ==========
app = FastAPI(title="SafeDig RL Environment")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
async def api_info():
    return {
        "name": "SafeDig RL Environment",
        "status": "running",
        "endpoints": {
            "reset": "/api/reset",
            "step": "/api/step",
            "state": "/api/state",
            "health": "/api/health",
            "info": "/api/info"
        }
    }

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/info")
async def info():
    return {
        "name": "SafeDig Environment",
        "version": "1.0.0",
        "actions": ["approve", "postpone", "scale_down", "mandate_safety"]
    }

@app.post("/api/reset")
async def reset(request: Request):
    try:
        body = await request.json()
        difficulty = body.get("difficulty", "easy")
    except:
        difficulty = "easy"
    obs = env.reset(difficulty=difficulty)
    return JSONResponse(content=obs.model_dump())

@app.post("/api/step")
async def step(request: Request):
    body = await request.json()
    decision = body.get("decision", "postpone")
    reasoning = body.get("reasoning", "")
    action = SafeDigAction(decision=decision, reasoning=reasoning)
    result = env.step(action)
    return JSONResponse(content=result.model_dump())

@app.get("/api/state")
async def state():
    return JSONResponse(content=env.state.model_dump())

# ========== GRADIO UI ==========
def format_sensors(obs: dict) -> str:
    if not obs:
        return "No data"
    md = "| Sensor | Value | Status |\n|--------|-------|--------|\n"
    co_status = '🔴 DANGER' if obs['gas_co_ppm'] > 70 else ('⚠️ CAUTION' if obs['gas_co_ppm'] > 35 else '✅ SAFE')
    h2s_status = '🔴 DANGER' if obs['gas_h2s_ppm'] > 20 else ('⚠️ CAUTION' if obs['gas_h2s_ppm'] > 5 else '✅ SAFE')
    methane_status = '🔴 DANGER' if obs['methane_pct'] > 1.5 else ('⚠️ CAUTION' if obs['methane_pct'] > 0.5 else '✅ SAFE')
    roof_status = '🔴 DANGER' if obs['roof_stability'] < 0.4 else ('⚠️ CAUTION' if obs['roof_stability'] < 0.7 else '✅ SAFE')
    quake_status = '🔴 DANGER' if obs['earthquake_risk'] > 0.6 else ('⚠️ CAUTION' if obs['earthquake_risk'] > 0.2 else '✅ SAFE')

    md += f"| CO Gas | {obs['gas_co_ppm']:.1f} ppm | {co_status} |\n"
    md += f"| H2S Gas | {obs['gas_h2s_ppm']:.1f} ppm | {h2s_status} |\n"
    md += f"| Methane | {obs['methane_pct']:.2f}% | {methane_status} |\n"
    md += f"| Roof Stability | {obs['roof_stability']:.2f} | {roof_status} |\n"
    md += f"| Earthquake Risk | {obs['earthquake_risk']:.2f} | {quake_status} |\n"
    md += f"| Ventilation | {'ON ✅' if obs['ventilation_on'] else 'OFF ❌'} | {'✅' if obs['ventilation_on'] else '🔴'} |\n"
    md += f"| Support Beams | {'OK ✅' if obs['support_beams_ok'] else 'BAD ❌'} | {'✅' if obs['support_beams_ok'] else '🔴'} |\n"
    md += f"| Last Near Miss | {obs['last_near_miss_days']} days ago | {'⚠️' if obs['last_near_miss_days'] < 7 else '✅'} |\n"
    return md

current_obs = {}

def reset_ui(difficulty: str):
    global current_obs
    obs = env.reset(difficulty=difficulty)
    current_obs = obs.model_dump()
    return format_sensors(current_obs), 0.0, 0, 0, "Ready ✅"

def act_ui(decision: str, reasoning: str):
    global current_obs
    action = SafeDigAction(decision=decision, reasoning=reasoning)
    result = env.step(action)
    current_obs = result.model_dump()
    
    # Get the message from the result
    message = current_obs.get("message", "")
    
    return (
        format_sensors(current_obs),
        current_obs["reward"],
        env.state.step_count,
        env.state.accidents,
        f"{'✅ Done' if current_obs['done'] else '🔄 In Progress'} — {message}"
    )

# Create Gradio UI
with gr.Blocks(title="SafeDig RL Environment") as demo:
    gr.Markdown("# ⛏️ SafeDig: Mining Safety RL Environment")
    gr.Markdown("Real-time decision-making for hazardous mining operations.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Controls")
            difficulty = gr.Radio(
                ["easy", "medium", "hard"],
                label="Difficulty Level",
                value="easy"
            )
            init_btn = gr.Button("🔄 Initialize Environment", variant="primary", size="lg")

            gr.Markdown("### 🤖 Agent Decision")
            decision = gr.Radio(
                ["approve", "postpone", "scale_down", "mandate_safety"],
                label="Choose Action"
            )
            reasoning = gr.Textbox(
                label="Reasoning",
                placeholder="Optional: Explain the decision...",
                lines=2
            )
            action_btn = gr.Button("▶️ Execute Action", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### 📈 Live Sensor Data")
            sensor_display = gr.Markdown("*Initialize environment to see sensor readings*")

    with gr.Row():
        reward_display   = gr.Number(label="Last Reward (0–1)", value=0.0, interactive=False)
        steps_display    = gr.Number(label="Steps Taken", value=0, interactive=False)
        accidents_display = gr.Number(label="Accidents", value=0, interactive=False)
        status_display   = gr.Textbox(label="Status", value="Not initialized", interactive=False)

    init_btn.click(
        reset_ui,
        inputs=[difficulty],
        outputs=[sensor_display, reward_display, steps_display, accidents_display, status_display]
    )
    action_btn.click(
        act_ui,
        inputs=[decision, reasoning],
        outputs=[sensor_display, reward_display, steps_display, accidents_display, status_display]
    )

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()