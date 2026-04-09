"""
SafeDig RL Environment - Gradio Web UI
Interactive dashboard for mining safety decision-making
Run: python gradio_ui.py
"""

import os
import json
import random
import threading
from typing import Dict, Any, Optional

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    exit(1)

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
    exit(1)

# ============ BACKEND API (Embedded) ============
app = FastAPI(title="SafeDig Environment API")

class Action(BaseModel):
    decision: str
    reasoning: str = ""

class ResetRequest(BaseModel):
    difficulty: str = "easy"

# Global environment state
class EnvironmentState:
    def __init__(self):
        self.reset(difficulty="easy")
    
    def reset(self, difficulty: str = "easy"):
        """Reset environment with realistic sensor values"""
        # Base values that change with difficulty
        if difficulty == "easy":
            self.gas_co_ppm = random.uniform(5, 20)
            self.gas_h2s_ppm = random.uniform(2, 8)
            self.methane_pct = random.uniform(0.1, 0.5)
            self.roof_stability = random.uniform(0.7, 0.95)
            self.earthquake_risk = random.uniform(0.1, 0.3)
        elif difficulty == "medium":
            self.gas_co_ppm = random.uniform(15, 40)
            self.gas_h2s_ppm = random.uniform(8, 15)
            self.methane_pct = random.uniform(0.3, 1.0)
            self.roof_stability = random.uniform(0.5, 0.8)
            self.earthquake_risk = random.uniform(0.2, 0.5)
        else:  # hard
            self.gas_co_ppm = random.uniform(30, 70)
            self.gas_h2s_ppm = random.uniform(12, 25)
            self.methane_pct = random.uniform(0.8, 1.8)
            self.roof_stability = random.uniform(0.3, 0.6)
            self.earthquake_risk = random.uniform(0.4, 0.8)
        
        self.ventilation_on = True
        self.support_beams_ok = True
        self.last_near_miss_days = random.randint(1, 30)
        self.step_count = 0
        self.accidents = 0
        self.difficulty = difficulty
        self.total_reward = 0.0
        
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        """Return current observation"""
        return {
            "gas_co_ppm": round(self.gas_co_ppm, 2),
            "gas_h2s_ppm": round(self.gas_h2s_ppm, 2),
            "methane_pct": round(self.methane_pct, 2),
            "roof_stability": round(self.roof_stability, 2),
            "earthquake_risk": round(self.earthquake_risk, 2),
            "ventilation_on": self.ventilation_on,
            "support_beams_ok": self.support_beams_ok,
            "last_near_miss_days": self.last_near_miss_days,
            "step_count": self.step_count,
            "accidents": self.accidents,
            "difficulty": self.difficulty,
            "message": f"Environment ready - {self.difficulty.upper()} difficulty"
        }
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk (0-1)"""
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
            risks.append(0.8)
        if not self.support_beams_ok:
            risks.append(0.9)
        
        # Recent near miss increases risk
        if self.last_near_miss_days < 7:
            risks.append(0.6)
        
        return sum(risks) / len(risks) if risks else 0.5
    
    def step(self, action: Action) -> Dict[str, Any]:
        """Execute an action and return results"""
        risk_score = self.calculate_risk_score()
        action_name = action.decision.lower()
        accident_occurred = False
        reward = 0
        message = ""
        
        # Action outcomes based on risk and difficulty
        if action_name == "approve":
            if risk_score < 0.4:
                reward = 15
                message = "✅ Safe conditions approved. Operation proceeds smoothly."
            elif risk_score < 0.7:
                reward = 5
                message = "⚠️ Moderate risk approved. Some issues arise but manageable."
            else:
                # High risk - chance of accident
                accident_chance = risk_score * (1.5 if self.difficulty == "hard" else 1.0)
                if random.random() < accident_chance:
                    reward = -50
                    accident_occurred = True
                    message = "🚨 ACCIDENT! High-risk approval led to disaster."
                    self.accidents += 1
                else:
                    reward = -10
                    message = "❌ High-risk approval barely avoided accident, but had issues."
        
        elif action_name == "postpone":
            if risk_score > 0.6:
                reward = 20
                message = "✅ Excellent decision! Postponing avoided major risks."
            elif risk_score > 0.3:
                reward = 10
                message = "⚠️ Postponing was reasonable, but some productivity lost."
            else:
                reward = -5
                message = "❌ Unnecessary postponement wasted resources."
        
        elif action_name == "scale_down":
            if risk_score > 0.5:
                reward = 12
                message = "✅ Scaling down reduced risk while maintaining some productivity."
            else:
                reward = 5
                message = "ℹ️ Scaled down operations. Safe but less efficient."
        
        elif action_name == "mandate_safety":
            reward = 8 if risk_score > 0.4 else 3
            message = "🛡️ Additional safety measures implemented. Increased protection."
            # Safety measures improve conditions
            self.ventilation_on = True
            self.support_beams_ok = True
        
        else:
            reward = 0
            message = "Invalid action"
        
        # Update environment after action
        self.step_count += 1
        self.total_reward += reward
        
        # Dynamic environment changes
        self._update_environment_after_step()
        
        # Check if episode should end (after 20 steps or accident)
        done = (self.step_count >= 20) or (accident_occurred and self.accidents >= 3)
        
        if done:
            if self.step_count >= 20:
                message += " Episode complete! Good work."
            else:
                message += " Episode ended due to accidents."
        
        return {
            "reward": reward,
            "total_reward": self.total_reward,
            "message": message,
            "accident_occurred": accident_occurred,
            "done": done,
            "step_count": self.step_count,
            "accidents": self.accidents
        }
    
    def _update_environment_after_step(self):
        """Update sensor readings after each step"""
        # Gases change randomly
        self.gas_co_ppm += random.uniform(-5, 8)
        self.gas_h2s_ppm += random.uniform(-2, 5)
        self.methane_pct += random.uniform(-0.2, 0.3)
        
        # Structural integrity degrades slowly
        self.roof_stability -= random.uniform(0, 0.05)
        self.earthquake_risk += random.uniform(-0.1, 0.1)
        
        # Keep within bounds
        self.gas_co_ppm = max(0, min(100, self.gas_co_ppm))
        self.gas_h2s_ppm = max(0, min(50, self.gas_h2s_ppm))
        self.methane_pct = max(0, min(3.0, self.methane_pct))
        self.roof_stability = max(0, min(1.0, self.roof_stability))
        self.earthquake_risk = max(0, min(1.0, self.earthquake_risk))
        
        # Equipment can fail
        if random.random() < 0.05 and self.difficulty == "hard":
            self.ventilation_on = False
        if random.random() < 0.03:
            self.support_beams_ok = False
        
        # Near miss timer
        self.last_near_miss_days += 1

# Global instance
env = EnvironmentState()

# FastAPI endpoints (kept for compatibility but not used directly)
@app.post("/reset")
async def reset_endpoint(request: ResetRequest):
    """Reset the environment"""
    obs = env.reset(difficulty=request.difficulty)
    return obs

@app.get("/state")
async def state_endpoint():
    """Get current state"""
    return env.get_observation()

@app.post("/step")
async def step_endpoint(action: Action):
    """Execute an action"""
    result = env.step(action)
    return result

@app.get("/info")
async def info_endpoint():
    """Get environment info"""
    return {
        "name": "SafeDig Environment",
        "version": "1.0.0",
        "actions": ["approve", "postpone", "scale_down", "mandate_safety"]
    }

# ============ GRADIO UI ============
current_obs = None
episode_data = {
    "difficulty": "easy",
    "reward": 0.0,
    "steps": 0,
    "accidents": 0,
    "decisions": []
}

def format_sensors(obs):
    """Format sensor data as markdown table"""
    if not obs:
        return "No data - please initialize environment"
    
    thresholds = {
        "gas_co_ppm": (35, 70),
        "gas_h2s_ppm": (5, 20),
        "methane_pct": (0.5, 1.5),
        "roof_stability": (0.7, 0.4),
        "earthquake_risk": (0.2, 0.6),
    }
    
    def get_status(key, value):
        if key not in thresholds:
            return "ℹ️"
        
        if key == "roof_stability":
            safe, danger = thresholds[key]
            if value > safe:
                return "✅ Safe"
            elif value < danger:
                return "🔴 Danger"
            else:
                return "⚠️ Caution"
        else:
            safe, danger = thresholds[key]
            if value <= safe:
                return "✅ Safe"
            elif value >= danger:
                return "🔴 Danger"
            else:
                return "⚠️ Caution"
    
    sensors = [
        ("CO Gas (ppm)", obs.get("gas_co_ppm", 0)),
        ("H2S Gas (ppm)", obs.get("gas_h2s_ppm", 0)),
        ("Methane (%)", obs.get("methane_pct", 0)),
        ("Roof Stability (0-1)", obs.get("roof_stability", 0)),
        ("Earthquake Risk (0-1)", obs.get("earthquake_risk", 0)),
    ]
    
    md = "| Sensor | Value | Status |\n|--------|-------|--------|\n"
    for name, value in sensors:
        key = name.split("(")[0].strip().lower().replace(" ", "_")
        if key == "co":
            key = "gas_co_ppm"
        elif key == "h2s":
            key = "gas_h2s_ppm"
        elif key == "methane":
            key = "methane_pct"
        elif key == "roof":
            key = "roof_stability"
        elif key == "earthquake":
            key = "earthquake_risk"
        
        status = get_status(key, value)
        if isinstance(value, float):
            md += f"| {name} | {value:.2f} | {status} |\n"
        else:
            md += f"| {name} | {value} | {status} |\n"
    
    # Equipment status
    ventilation_status = "ON ✅" if obs.get("ventilation_on") else "OFF ❌"
    beams_status = "OK ✅" if obs.get("support_beams_ok") else "BAD ❌"
    near_miss = obs.get("last_near_miss_days", 0)
    
    md += f"| Ventilation | {ventilation_status} | {'✅' if obs.get('ventilation_on') else '🔴'} |\n"
    md += f"| Support Beams | {beams_status} | {'✅' if obs.get('support_beams_ok') else '🔴'} |\n"
    md += f"| Last Near Miss | {near_miss} days ago | {'⚠️' if near_miss < 7 else '✅'} |\n"
    
    return md

def reset_environment(difficulty: str):
    """Reset the environment and return observation"""
    global current_obs, episode_data
    
    try:
        # Use the embedded backend directly
        obs = env.reset(difficulty=difficulty)
        current_obs = obs
        
        episode_data = {
            "difficulty": difficulty,
            "reward": 0.0,
            "steps": 0,
            "accidents": 0,
            "decisions": []
        }
        
        sensors_md = format_sensors(current_obs)
        message = f"✅ Environment initialized with **{difficulty.upper()}** difficulty\n\n{current_obs.get('message', 'Ready')}"
        
        return (
            message,
            sensors_md,
            float(0.0),
            int(0),
            int(0),
            "Ready"
        )
    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            "Failed to initialize",
            float(0.0),
            int(0),
            int(0),
            "Error"
        )

def take_action(decision, reasoning=""):
    """Take an action in the environment"""
    global current_obs, episode_data
    
    if current_obs is None:
        return (
            "❌ Please initialize environment first",
            "No data - please initialize environment",
            float(0.0),
            int(0),
            int(0),
            "Not initialized"
        )
    
    try:
        # Use embedded backend
        action = Action(decision=decision, reasoning=reasoning or f"Agent decided to {decision}")
        result = env.step(action)
        
        # Get updated observation
        current_obs = env.get_observation()
        
        # Update episode data
        episode_data["reward"] = result.get("total_reward", 0.0)
        episode_data["steps"] = result.get("step_count", 0)
        episode_data["accidents"] = result.get("accidents", 0)
        episode_data["decisions"].append({
            "action": decision,
            "reward": result.get("reward", 0.0),
            "message": result.get("message", "")
        })
        
        # Format message
        message = f"**Decision: {decision.upper()}**\n\n{result.get('message', 'Action executed')}"
        
        if result.get("accident_occurred"):
            message += "\n\n🚨 **ACCIDENT OCCURRED!**"
        
        if result.get("done"):
            message += "\n\n🏁 **Episode Complete!**"
        
        reward = float(result.get("total_reward", 0.0))
        
        return (
            message,
            format_sensors(current_obs),
            reward,
            int(result.get("step_count", 0)),
            int(result.get("accidents", 0)),
            "Complete" if result.get("done") else "In Progress"
        )
    
    except Exception as e:
        return (
            f"❌ Error taking action: {str(e)}",
            format_sensors(current_obs),
            float(episode_data["reward"]),
            int(episode_data["steps"]),
            int(episode_data["accidents"]),
            "Error"
        )

def run_baseline_agent(difficulty):
    """Run a simple rule-based baseline agent (no OpenAI required)"""
    global current_obs, episode_data
    
    try:
        # First ensure environment is initialized
        obs = env.reset(difficulty=difficulty)
        current_obs = obs
        
        # Reset episode data
        episode_data = {
            "difficulty": difficulty,
            "reward": 0.0,
            "steps": 0,
            "accidents": 0,
            "decisions": []
        }
        
        # Simple rule-based decision making
        risk_factors = []
        
        if obs.get("gas_co_ppm", 0) > 50:
            risk_factors.append("high CO")
        if obs.get("gas_h2s_ppm", 0) > 15:
            risk_factors.append("high H2S")
        if obs.get("methane_pct", 0) > 1.2:
            risk_factors.append("high methane")
        if obs.get("roof_stability", 1) < 0.5:
            risk_factors.append("unstable roof")
        if obs.get("earthquake_risk", 0) > 0.5:
            risk_factors.append("high quake risk")
        if not obs.get("ventilation_on", True):
            risk_factors.append("ventilation off")
        if not obs.get("support_beams_ok", True):
            risk_factors.append("bad beams")
        
        # Decision logic
        if len(risk_factors) >= 3:
            decision = "postpone"
            reasoning = f"Multiple risks detected: {', '.join(risk_factors[:3])}. Postponing for safety."
        elif len(risk_factors) >= 1:
            decision = "mandate_safety"
            reasoning = f"Risks detected: {', '.join(risk_factors)}. Adding safety measures."
        else:
            decision = "approve"
            reasoning = "All conditions within safe limits. Proceeding with approval."
        
        # Execute the action
        action = Action(decision=decision, reasoning=reasoning)
        result = env.step(action)
        current_obs = env.get_observation()
        
        # Update episode data
        episode_data["reward"] = result.get("total_reward", 0.0)
        episode_data["steps"] = result.get("step_count", 0)
        episode_data["accidents"] = result.get("accidents", 0)
        
        # Format the result message
        result_text = f"""✅ **Baseline Agent Decision: {decision.upper()}**

**Reasoning:** {reasoning}

**Outcome:** {result.get('message', 'Action executed')}

**Reward:** {result.get('reward', 0):.1f}
**Total Reward:** {result.get('total_reward', 0):.1f}
**Steps:** {result.get('step_count', 0)}
**Accidents:** {result.get('accidents', 0)}"""
        
        # Also update the UI displays by returning the values
        return (
            result_text,  # baseline_result
            f"✅ Environment initialized with **{difficulty.upper()}** difficulty\n\n{current_obs.get('message', 'Ready')}",  # result_display
            format_sensors(current_obs),  # sensor_display
            float(result.get("total_reward", 0.0)),  # reward_display
            int(result.get("step_count", 0)),  # steps_display
            int(result.get("accidents", 0)),  # accidents_display
            "In Progress" if not result.get("done") else "Complete"  # status_display
        )
    
    except Exception as e:
        error_msg = f"❌ Baseline agent error: {str(e)}"
        return (
            error_msg,  # baseline_result
            error_msg,  # result_display
            "Error occurred",  # sensor_display
            float(0.0),  # reward_display
            int(0),  # steps_display
            int(0),  # accidents_display
            "Error"  # status_display
        )

# Build Gradio interface with DARK MODE and repositioned layout
with gr.Blocks(
    title="SafeDig RL Environment",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { 
        max-width: 1200px; 
        margin: auto;
        background-color: #1a1a1a !important;
    }
    body, .gradio-container, .gr-box, .gr-form, .gr-panel {
        background-color: #1a1a1a !important;
    }
    .gr-box, .gr-form, .gr-panel, .main {
        background-color: #1a1a1a !important;
    }
    label, span, div, p, h1, h2, h3, h4, h5, .markdown-text {
        color: #ffffff !important;
    }
    table, th, td {
        color: #ffffff !important;
        border-color: #ffffff !important;
        background-color: #2a2a2a !important;
    }
    input, textarea, select {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    .gr-button {
        background-color: #3a3a3a !important;
    }
    .gr-button-primary {
        background-color: #2563eb !important;
    }
    """
) as demo:
    
    gr.Markdown("# ⛏️ SafeDig: Mining Safety RL Environment")
    gr.Markdown("Real-time decision-making for hazardous mining operations. Help the agent learn to make safe choices.")
    
    # Row 1: Controls (Left) and Live Sensor Data (Right)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Controls")
            difficulty = gr.Radio(
                ["easy", "medium", "hard"],
                value="easy",
                label="Difficulty Level",
                info="Select scenario complexity"
            )
            init_btn = gr.Button("🔄 Initialize Environment", variant="primary", size="lg")
            
            gr.Markdown("### 🤖 Agent Decision")
            decision = gr.Radio(
                ["approve", "postpone", "scale_down", "mandate_safety"],
                label="Choose Action",
                info="What should the agent do?"
            )
            reasoning = gr.Textbox(
                label="Reasoning",
                placeholder="Optional: Explain the decision...",
                lines=2
            )
            action_btn = gr.Button("▶️ Execute Action", variant="primary", size="lg")
            result_display = gr.Markdown("*Result will appear here*")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Live Sensor Data")
            sensor_display = gr.Markdown("*Initialize environment to see sensor readings*")
    
    # Row 2: Episode Metrics (below Execute Action button)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Episode Metrics")
            reward_display = gr.Number(label="Cumulative Reward", value=0.0, interactive=False)
            steps_display = gr.Number(label="Steps Taken", value=0, interactive=False)
            accidents_display = gr.Number(label="Accidents", value=0, interactive=False)
            status_display = gr.Textbox(label="Status", value="Not initialized", interactive=False)
    
    # Row 3: Baseline Agent
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Run Baseline Agent")
            baseline_difficulty = gr.Radio(
                ["easy", "medium", "hard"],
                value="easy",
                label="Baseline Difficulty",
                info="Let the rule-based agent make decisions"
            )
            baseline_btn = gr.Button("🚀 Run Baseline Agent", variant="secondary")
            baseline_result = gr.Markdown()
    
    # Event handlers
    init_btn.click(
        reset_environment,
        inputs=difficulty,
        outputs=[result_display, sensor_display, reward_display, steps_display, accidents_display, status_display]
    )
    
    action_btn.click(
        take_action,
        inputs=[decision, reasoning],
        outputs=[result_display, sensor_display, reward_display, steps_display, accidents_display, status_display]
    )
    
    baseline_btn.click(
        run_baseline_agent,
        inputs=baseline_difficulty,
        outputs=[baseline_result, result_display, sensor_display, reward_display, steps_display, accidents_display, status_display]
    )
    
    gr.Markdown("""
    ---
    ### 📚 How It Works
    
    **Actions & Rewards:**
    - **Approve** (+15 safe, -50 accident): Full task approval
    - **Postpone** (+20 high risk, -5 safe): Delay the task
    - **Scale Down** (+12 moderate risk, +5 safe): Reduce scope
    - **Mandate Safety** (+8 high risk, +3 safe): Add protective measures
    
    **Risk Factors Monitored:**
    - CO Gas (danger > 70 ppm)
    - H2S Gas (danger > 20 ppm)
    - Methane (danger > 1.5%)
    - Roof Stability (danger < 0.4)
    - Earthquake Risk (danger > 0.6)
    - Equipment status (ventilation, support beams)
    
    **Difficulty Levels:**
    - **Easy**: Lower initial risks, slower degradation
    - **Medium**: Mixed conditions requiring analysis
    - **Hard**: Higher initial risks, faster degradation
    
    ---
    🏆 **OpenEnv Framework** | Mining Safety RL Benchmark
    """)

# ============ MAIN ============
if __name__ == "__main__":
    # Start Gradio only (FastAPI is embedded but not separately served)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )