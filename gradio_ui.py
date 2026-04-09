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
        else:  # hard
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

    def calculate_risk_score(self):
        """Calculate current risk level (0-1)"""
        risks = []
        
        # Gas risks
        if self.gas_co_ppm > 70:
            risks.append(1.0)
        elif self.gas_co_ppm > 35:
            risks.append(0.7)
        else:
            risks.append(0.2)
        
        if self.gas_h2s_ppm > 20:
            risks.append(1.0)
        elif self.gas_h2s_ppm > 5:
            risks.append(0.7)
        else:
            risks.append(0.2)
        
        if self.methane_pct > 1.5:
            risks.append(1.0)
        elif self.methane_pct > 0.5:
            risks.append(0.7)
        else:
            risks.append(0.2)
        
        # Structural risks
        if self.roof_stability < 0.4:
            risks.append(1.0)
        elif self.roof_stability < 0.7:
            risks.append(0.7)
        else:
            risks.append(0.2)
        
        if self.earthquake_risk > 0.6:
            risks.append(1.0)
        elif self.earthquake_risk > 0.2:
            risks.append(0.7)
        else:
            risks.append(0.2)
        
        # Equipment status
        if not self.ventilation_on:
            risks.append(0.9)
        if not self.support_beams_ok:
            risks.append(0.8)
        
        return sum(risks) / len(risks) if risks else 0.5

    def step(self, action):
        action_map = ["approve", "postpone", "scale_down", "mandate_safety"]
        action_name = action_map[action] if 0 <= action < 4 else "approve"
        
        # Calculate current risk level
        risk_score = self.calculate_risk_score()
        
        reward = 0
        accident_occurred = False
        message = ""
        
        # Risk-based rewards
        if action_name == "approve":
            if risk_score < 0.3:  # Very safe
                reward = 20
                message = "✅ Safe approval! Maximum productivity."
            elif risk_score < 0.6:  # Moderate risk
                reward = 5
                message = "⚠️ Moderate risk approval. Some issues but manageable."
            else:  # High risk
                # Chance of accident based on difficulty
                accident_chance = risk_score * (1.5 if self.difficulty == "hard" else 1.0)
                if random.random() < accident_chance:
                    reward = -100
                    accident_occurred = True
                    self.accidents += 1
                    message = "💥 ACCIDENT! High-risk approval caused disaster!"
                else:
                    reward = -20
                    message = "❌ Lucky escape! High-risk approval barely avoided accident."
        
        elif action_name == "postpone":
            if risk_score > 0.6:  # High risk - good decision
                reward = 30
                message = "✅ Excellent! Postponing avoided major risks."
            elif risk_score > 0.3:  # Moderate risk
                reward = 15
                message = "⚠️ Reasonable postponement. Some productivity loss."
            else:  # Low risk - unnecessary
                reward = -10
                message = "❌ Unnecessary postponement. Wasted resources."
        
        elif action_name == "scale_down":
            if risk_score > 0.5:  # Good for high risk
                reward = 20
                message = "✅ Smart! Scaling down reduced risk significantly."
            elif risk_score > 0.2:
                reward = 10
                message = "ℹ️ Scaling down was cautious but reduced productivity."
            else:
                reward = 0
                message = "⚠️ Scaling down unnecessary for safe conditions."
        
        elif action_name == "mandate_safety":
            if risk_score > 0.4:
                reward = 15
                message = "🛡️ Good! Safety measures increased protection."
                # Safety measures improve conditions
                self.ventilation_on = True
                self.support_beams_ok = True
            else:
                reward = 5
                message = "✅ Safety measures added. Minor productivity loss."
        
        self.total_reward += reward
        self.step_count += 1
        
        # Update environment values (risks increase over time)
        self.gas_co_ppm = round(max(0, min(100, self.gas_co_ppm + random.uniform(-3, 10))), 2)
        self.gas_h2s_ppm = round(max(0, min(50, self.gas_h2s_ppm + random.uniform(-2, 8))), 2)
        self.methane_pct = round(max(0, min(3.0, self.methane_pct + random.uniform(-0.1, 0.4))), 2)
        self.roof_stability = round(max(0, min(1.0, self.roof_stability - random.uniform(0, 0.08))), 2)
        self.earthquake_risk = round(max(0, min(1.0, self.earthquake_risk + random.uniform(-0.05, 0.05))), 2)
        
        # Equipment can fail in hard difficulty
        if self.difficulty == "hard" and random.random() < 0.1:
            self.ventilation_on = False
        if random.random() < 0.05:
            self.support_beams_ok = False
        
        done = self.step_count >= 20 or self.accidents >= 3
        
        return {
            "observation": self.get_obs(),
            "reward": reward,
            "done": done,
            "info": {"message": message, "accident_occurred": accident_occurred}
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
    md = "| Sensor | Value | Status |\n|--------|-------|--------|\n"
    
    # Gas sensors with status
    co_status = "🔴 DANGER" if obs['gas_co_ppm'] > 70 else ("⚠️ CAUTION" if obs['gas_co_ppm'] > 35 else "✅ SAFE")
    h2s_status = "🔴 DANGER" if obs['gas_h2s_ppm'] > 20 else ("⚠️ CAUTION" if obs['gas_h2s_ppm'] > 5 else "✅ SAFE")
    methane_status = "🔴 DANGER" if obs['methane_pct'] > 1.5 else ("⚠️ CAUTION" if obs['methane_pct'] > 0.5 else "✅ SAFE")
    roof_status = "🔴 DANGER" if obs['roof_stability'] < 0.4 else ("⚠️ CAUTION" if obs['roof_stability'] < 0.7 else "✅ SAFE")
    quake_status = "🔴 DANGER" if obs['earthquake_risk'] > 0.6 else ("⚠️ CAUTION" if obs['earthquake_risk'] > 0.2 else "✅ SAFE")
    
    md += f"| CO Gas | {obs['gas_co_ppm']} ppm | {co_status} |\n"
    md += f"| H2S Gas | {obs['gas_h2s_ppm']} ppm | {h2s_status} |\n"
    md += f"| Methane | {obs['methane_pct']}% | {methane_status} |\n"
    md += f"| Roof Stability | {obs['roof_stability']} | {roof_status} |\n"
    md += f"| Earthquake Risk | {obs['earthquake_risk']} | {quake_status} |\n"
    md += f"| Ventilation | {'ON ✅' if obs['ventilation_on'] else 'OFF ❌'} | {'✅' if obs['ventilation_on'] else '🔴'} |\n"
    md += f"| Support Beams | {'OK ✅' if obs['support_beams_ok'] else 'BAD ❌'} | {'✅' if obs['support_beams_ok'] else '🔴'} |\n"
    md += f"| Last Near Miss | {obs['last_near_miss_days']} days ago | {'⚠️' if obs['last_near_miss_days'] < 7 else '✅'} |\n"
    
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
    
    # Add accident message to output
    message = result['info'].get('message', '')
    if result['info'].get('accident_occurred'):
        message = f"💥 {message}"
    
    return (
        format_sensors(current_obs),
        result["reward"],
        current_obs["step_count"],
        current_obs["accidents"],
        "Complete" if result["done"] else f"In Progress - {message}"
    )

with gr.Blocks(title="SafeDig RL Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⛏️ SafeDig: Mining Safety RL Environment")
    gr.Markdown("Real-time decision-making for hazardous mining operations. Help the agent learn to make safe choices.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Controls")
            difficulty = gr.Radio(["easy", "medium", "hard"], label="Difficulty Level", value="easy")
            init_btn = gr.Button("🔄 Initialize Environment", variant="primary", size="lg")
            
            gr.Markdown("### 🤖 Agent Decision")
            decision = gr.Radio(["approve", "postpone", "scale_down", "mandate_safety"], label="Choose Action")
            reasoning = gr.Textbox(label="Reasoning", placeholder="Optional: Explain the decision...", lines=2)
            action_btn = gr.Button("▶️ Execute Action", variant="primary", size="lg")
            result_display = gr.Markdown("*Result will appear here*")
        
        with gr.Column():
            gr.Markdown("### 📈 Live Sensor Data")
            sensor_display = gr.Markdown("*Initialize environment to see sensor readings*")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Episode Metrics")
            reward_display = gr.Number(label="Cumulative Reward", value=0.0, interactive=False)
            steps_display = gr.Number(label="Steps Taken", value=0, interactive=False)
            accidents_display = gr.Number(label="Accidents", value=0, interactive=False)
            status_display = gr.Textbox(label="Status", value="Not initialized", interactive=False)

    init_btn.click(reset_ui, [difficulty], [sensor_display, reward_display, steps_display, accidents_display, status_display])
    action_btn.click(act_ui, [decision, reasoning], [sensor_display, reward_display, steps_display, accidents_display, status_display])

# Mount Gradio on the same FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    """Entry point for the server script"""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()