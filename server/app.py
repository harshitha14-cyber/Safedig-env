import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from environment import SafeDigEnvironment, SensorReliabilityEnvironment, RescueEnvironment
from models import SafeDigAction, SensorReliabilityAction, RescueAction

# ── Environment instances ────────────────────────────────────────────────
envs = {
    "safety_decision": SafeDigEnvironment(),
    "sensor_reliability": SensorReliabilityEnvironment(),
    "rescue_coordination": RescueEnvironment(),
}

# ── FastAPI ──────────────────────────────────────────────────────────────
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
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "health": "/health",
            "info": "/info"
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
        "tasks": list(envs.keys()),
        "actions": {
            "safety_decision": ["approve", "postpone", "scale_down", "mandate_safety"],
            "sensor_reliability": ["trust_and_proceed", "request_recalibration", "cross_reference", "emergency_stop"],
            "rescue_coordination": ["deploy_robotic_drill", "manual_extraction", "seal_section", "call_for_support"],
        }
    }

@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
        task = body.get("task", "safety_decision")
        difficulty = body.get("difficulty", "easy")
    except:
        task = "safety_decision"
        difficulty = "easy"
    
    env = envs.get(task, envs["safety_decision"])
    obs = env.reset(difficulty=difficulty)
    return JSONResponse(content=obs.model_dump())

@app.post("/step")
async def step(request: Request):
    body = await request.json()
    task = body.get("task", "safety_decision")
    decision = body.get("decision", "postpone")
    reasoning = body.get("reasoning", "")
    
    if task == "safety_decision":
        action = SafeDigAction(decision=decision, reasoning=reasoning)  # type: ignore
    elif task == "sensor_reliability":
        action = SensorReliabilityAction(decision=decision, reasoning=reasoning)  # type: ignore
    elif task == "rescue_coordination":
        action = RescueAction(decision=decision, reasoning=reasoning)  # type: ignore
    else:
        action = SafeDigAction(decision=decision, reasoning=reasoning)  # type: ignore
    
    env = envs.get(task, envs["safety_decision"])
    result = env.step(action)
    return JSONResponse(content=result.model_dump())

@app.get("/state")
async def state(task: str = "safety_decision"):
    env = envs.get(task, envs["safety_decision"])
    return JSONResponse(content=env.state.model_dump())

# ── Gradio UI ──────────────────────────────────────────────��─────────────

def format_task1(obs: dict) -> str:
    """Safety Decision formatting"""
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

def format_task2(obs: dict) -> str:
    """Sensor Reliability formatting"""
    if not obs:
        return "No data"
    md = "| Metric | Value | Status |\n|--------|-------|--------|\n"
    
    # Determine status based on readings
    conflict = "⚠️ YES" if obs.get('conflict_detected', False) else "✅ NO"
    
    # Confidence score status
    conf_score = obs.get('confidence_score', 0.5)
    if conf_score < 0.4:
        conf_icon = "🔴 LOW"
    elif conf_score < 0.7:
        conf_icon = "⚠️ MEDIUM"
    else:
        conf_icon = "✅ HIGH"
    
    # Sensor age status
    sensor_age = obs.get('sensor_age_days', 0)
    if sensor_age > 20:
        age_icon = "🔴 OLD"
    elif sensor_age > 10:
        age_icon = "⚠️ AGING"
    else:
        age_icon = "✅ FRESH"
    
    # Ventilation status
    vent = "✅ ON" if obs.get('ventilation_on', False) else "❌ OFF"
    
    # Primary vs Secondary CO comparison
    primary_co = obs.get('primary_co_ppm', 0)
    secondary_co = obs.get('secondary_co_ppm', 0)
    co_diff = abs(primary_co - secondary_co)
    
    if co_diff > 20:
        diff_icon = "🔴 HIGH CONFLICT"
    elif co_diff > 10:
        diff_icon = "⚠️ MODERATE CONFLICT"
    else:
        diff_icon = "✅ AGREE"
    
    md += f"| Primary CO | {primary_co:.1f} ppm | - |\n"
    md += f"| Secondary CO | {secondary_co:.1f} ppm | - |\n"
    md += f"| CO Readings | {diff_icon} | - |\n"
    md += f"| Conflict Detected | {conflict} | - |\n"
    md += f"| Confidence | {conf_icon} ({conf_score:.2f}) | - |\n"
    md += f"| Sensor Age | {age_icon} ({sensor_age} days) | - |\n"
    md += f"| Ventilation | {vent} | - |\n"
    
    return md

def format_task3(obs: dict) -> str:
    """Rescue Coordination formatting"""
    if not obs:
        return "No data"
    md = "| Parameter | Value | Status |\n|-----------|-------|--------|\n"
    ox = "🔴" if obs.get('oxygen_remaining_pct', 0.5) < 0.3 else ("⚠️" if obs.get('oxygen_remaining_pct', 0.5) < 0.6 else "✅")
    ob = "🔴" if obs.get('path_obstruction', 0) > 0.6 else ("⚠️" if obs.get('path_obstruction', 0) > 0.3 else "✅")
    sr = "🔴" if obs.get('structural_risk', 0) > 0.6 else ("⚠️" if obs.get('structural_risk', 0) > 0.3 else "✅")
    
    md += f"| Trapped Personnel | {obs.get('trapped_personnel', 0)} | 🆘 |\n"
    md += f"| Oxygen Remaining | {obs.get('oxygen_remaining_pct', 0.5)*100:.0f}% | {ox} |\n"
    md += f"| Path Obstruction | {obs.get('path_obstruction', 0):.2f} | {ob} |\n"
    md += f"| Structural Risk | {obs.get('structural_risk', 0):.2f} | {sr} |\n"
    md += f"| Resources Available | {obs.get('available_resources', 0)} units | - |\n"
    md += f"| Time Elapsed | {obs.get('time_elapsed_minutes', 0)} min | ⏱️ |\n"
    return md

TASKS = {
    "⛏️ Safety Decision": {
        "key": "safety_decision",
        "actions": ["approve", "postpone", "scale_down", "mandate_safety"],
        "formatter": format_task1
    },
    "🔍 Sensor Reliability": {
        "key": "sensor_reliability",
        "actions": ["trust_and_proceed", "request_recalibration", "cross_reference", "emergency_stop"],
        "formatter": format_task2
    },
    "🚨 Rescue Coordination": {
        "key": "rescue_coordination",
        "actions": ["deploy_robotic_drill", "manual_extraction", "seal_section", "call_for_support"],
        "formatter": format_task3
    },
}

current_obs = {}
current_task = "safety_decision"

def reset_ui(task_label: str, difficulty: str):
    global current_obs, current_task
    current_task = TASKS[task_label]["key"]
    env = envs[current_task]
    obs = env.reset(difficulty=difficulty)
    current_obs = obs.model_dump()
    formatter = TASKS[task_label]["formatter"]
    
    return (
        formatter(current_obs),
        gr.update(choices=TASKS[task_label]["actions"], value=TASKS[task_label]["actions"][0]),
        0.0,
        0,
        0,
        "Ready ✅"
    )

def act_ui(task_label: str, decision: str, reasoning: str):
    global current_obs, current_task
    current_task = TASKS[task_label]["key"]
    env = envs[current_task]
    
    if current_task == "safety_decision":
        action = SafeDigAction(decision=decision, reasoning=reasoning)  # type: ignore
    elif current_task == "sensor_reliability":
        action = SensorReliabilityAction(decision=decision, reasoning=reasoning)  # type: ignore
    else:
        action = RescueAction(decision=decision, reasoning=reasoning)  # type: ignore
    
    result = env.step(action)
    current_obs = result.model_dump()
    formatter = TASKS[task_label]["formatter"]
    message = current_obs.get("message", "")
    
    return (
        formatter(current_obs),
        current_obs["reward"],
        env.state.step_count,
        getattr(env.state, "accidents", getattr(env.state, "casualties", 0)),
        f"{'✅ Done' if current_obs['done'] else '🔄 In Progress'} — {message}"
    )

def update_actions(task_label):
    return gr.update(choices=TASKS[task_label]["actions"], value=TASKS[task_label]["actions"][0])

# Create Gradio UI
with gr.Blocks(title="SafeDig RL Environment") as demo:
    gr.Markdown("# ⛏️ SafeDig: Mining Safety RL Environment")
    gr.Markdown("Three critical tasks: **Safety Decision** | **Sensor Reliability** | **Rescue Coordination**")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Controls")
            task_radio = gr.Radio(
                choices=list(TASKS.keys()),
                label="Select Task",
                value="⛏️ Safety Decision"
            )
            difficulty = gr.Radio(
                choices=["easy", "medium", "hard"],
                label="Difficulty Level",
                value="easy"
            )
            init_btn = gr.Button("🔄 Initialize Environment", variant="primary", size="lg")

            gr.Markdown("### 🤖 Agent Decision")
            decision = gr.Radio(
                choices=TASKS["⛏️ Safety Decision"]["actions"],
                label="Choose Action",
                value=TASKS["⛏️ Safety Decision"]["actions"][0]
            )
            reasoning = gr.Textbox(
                label="Reasoning",
                placeholder="Optional: Explain the decision...",
                lines=2
            )
            action_btn = gr.Button("▶️ Execute Action", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### 📈 Live Data")
            sensor_display = gr.Markdown("*Initialize environment to see sensor readings*")

    with gr.Row():
        reward_display = gr.Number(label="Last Reward (0–1)", value=0.0, interactive=False)
        steps_display = gr.Number(label="Steps Taken", value=0, interactive=False)
        incidents_display = gr.Number(label="Incidents/Casualties", value=0, interactive=False)
        status_display = gr.Textbox(label="Status", value="Not initialized", interactive=False)

    task_radio.change(update_actions, inputs=[task_radio], outputs=[decision])
    
    init_btn.click(
        reset_ui,
        inputs=[task_radio, difficulty],
        outputs=[sensor_display, decision, reward_display, steps_display, incidents_display, status_display]
    )
    action_btn.click(
        act_ui,
        inputs=[task_radio, decision, reasoning],
        outputs=[sensor_display, reward_display, steps_display, incidents_display, status_display]
    )

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="127.0.0.1", port=7860)

if __name__ == "__main__":
    main()