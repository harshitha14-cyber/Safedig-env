import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
import json

# ========== ENVIRONMENT CLASS ==========
class SafeDigEnv:
    def __init__(self):
        self.reset("easy")

    def reset(self, difficulty="easy"):
        if difficulty == "easy":
            self.gas_co_ppm = round(random.uniform(5, 20), 2)
            self.gas_h2s_ppm = round(random.uniform(2, 8), 2)
            self.methane_pct = round(random.uniform(0.1, 0.5), 2)
            self.roof_stability = round(random.uniform(0.7, 0.95), 2)
            self.earthquake_risk = round(random.uniform(0.1, 0.3), 2)
        elif difficulty == "medium":
            self.gas_co_ppm = round(random.uniform(15, 40), 2)
            self.gas_h2s_ppm = round(random.uniform(8, 15), 2)
            self.methane_pct = round(random.uniform(0.3, 1.0), 2)
            self.roof_stability = round(random.uniform(0.5, 0.8), 2)
            self.earthquake_risk = round(random.uniform(0.2, 0.5), 2)
        else:
            self.gas_co_ppm = round(random.uniform(30, 70), 2)
            self.gas_h2s_ppm = round(random.uniform(12, 25), 2)
            self.methane_pct = round(random.uniform(0.8, 1.8), 2)
            self.roof_stability = round(random.uniform(0.3, 0.6), 2)
            self.earthquake_risk = round(random.uniform(0.4, 0.8), 2)

        self.ventilation_on = True
        self.support_beams_ok = True
        self.last_near_miss_days = random.randint(1, 30)
        self.step_count = 0
        self.accidents = 0
        self.difficulty = difficulty
        self.total_reward = 0.0

        return self.get_obs()

    def get_obs(self):
        return {
            "gas_co_ppm": self.gas_co_ppm,
            "gas_h2s_ppm": self.gas_h2s_ppm,
            "methane_pct": self.methane_pct,
            "roof_stability": self.roof_stability,
            "earthquake_risk": self.earthquake_risk,
            "ventilation_on": self.ventilation_on,
            "support_beams_ok": self.support_beams_ok,
            "last_near_miss_days": self.last_near_miss_days,
            "step_count": self.step_count,
            "accidents": self.accidents,
            "difficulty": self.difficulty,
        }

    def step(self, action):
        action_map = ["approve", "postpone", "scale_down", "mandate_safety"]
        action_name = action_map[action] if 0 <= action < 4 else "approve"

        # Simple reward logic
        reward = 10 if action == 0 else 5
        self.total_reward += reward
        self.step_count += 1

        # Update values randomly
        self.gas_co_ppm = round(max(0, min(100, self.gas_co_ppm + random.uniform(-5, 8))), 2)
        self.gas_h2s_ppm = round(max(0, min(50, self.gas_h2s_ppm + random.uniform(-2, 5))), 2)
        self.methane_pct = round(max(0, min(3.0, self.methane_pct + random.uniform(-0.2, 0.3))), 2)

        done = self.step_count >= 20

        return {
            "observation": self.get_obs(),
            "reward": reward,
            "done": done,
            "info": {"message": f"Executed {action_name}"}
        }

env = SafeDigEnv()

# ========== FASTAPI APP ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/info")
async def info():
    return {
        "name": "SafeDig Environment",
        "version": "1.0.0",
        "actions": ["approve", "postpone", "scale_down", "mandate_safety"]
    }

@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
        difficulty = body.get("difficulty", "easy")
    except:
        difficulty = "easy"
    obs = env.reset(difficulty)
    return JSONResponse(content=obs)

@app.get("/state")
async def state():
    return JSONResponse(content=env.get_obs())

@app.post("/step")
async def step(request: Request):
    body = await request.json()
    action = body.get("action", 0)
    result = env.step(action)
    return JSONResponse(content=result)

# ========== GRADIO UI ==========
def format_sensors(obs):
    if not obs:
        return "No data"
    md = "| Sensor | Value |\n|--------|-------|\n"
    md += f"| CO Gas | {obs['gas_co_ppm']} ppm |\n"
    md += f"| H2S Gas | {obs['gas_h2s_ppm']} ppm |\n"
    md += f"| Methane | {obs['methane_pct']}% |\n"
    md += f"| Roof Stability | {obs['roof_stability']} |\n"
    md += f"| Earthquake Risk | {obs['earthquake_risk']} |\n"
    md += f"| Ventilation | {'ON' if obs['ventilation_on'] else 'OFF'} |\n"
    md += f"| Support Beams | {'OK' if obs['support_beams_ok'] else 'BAD'} |\n"
    return md

current_obs = None

def reset_ui(difficulty):
    global current_obs
    current_obs = env.reset(difficulty)
    return format_sensors(current_obs), 0, 0, 0, "Ready"

def act_ui(decision, reasoning):
    global current_obs
    action_map = {"approve": 0, "postpone": 1, "scale_down": 2, "mandate_safety": 3}
    result = env.step(action_map[decision])
    current_obs = result["observation"]
    return (
        format_sensors(current_obs),
        result["reward"],
        current_obs["step_count"],
        current_obs["accidents"],
        "Complete" if result["done"] else "In Progress"
    )

with gr.Blocks(title="SafeDig RL Environment") as demo:
    gr.Markdown("# ⛏️ SafeDig: Mining Safety RL Environment")
    with gr.Row():
        with gr.Column():
            difficulty = gr.Radio(["easy", "medium", "hard"], label="Difficulty")
            init_btn = gr.Button("Initialize Environment")
            decision = gr.Radio(["approve", "postpone", "scale_down", "mandate_safety"], label="Action")
            reasoning = gr.Textbox(label="Reasoning", lines=2)
            action_btn = gr.Button("Execute Action")
        with gr.Column():
            sensor_display = gr.Markdown("No data")
            reward_display = gr.Number(label="Reward")
            steps_display = gr.Number(label="Steps")
            accidents_display = gr.Number(label="Accidents")
            status_display = gr.Textbox(label="Status")

    init_btn.click(reset_ui, [difficulty], [sensor_display, reward_display, steps_display, accidents_display, status_display])
    action_btn.click(act_ui, [decision, reasoning], [sensor_display, reward_display, steps_display, accidents_display, status_display])

# Mount Gradio on the same FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)