"""
SafeDig RL Environment - Gradio Web UI
Interactive dashboard for mining safety decision-making
"""

import os
import json
import requests
import gradio as gr

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

current_obs = None
episode_data = {
    "difficulty": "easy",
    "reward": 0.0,
    "steps": 0,
    "accidents": 0,
    "decisions": []
}

def format_sensors(obs):
    if not obs:
        return "No data - please initialize environment"
    
    md = "| Sensor | Value | Status |\n|--------|-------|--------|\n"
    
    sensors = [
        ("CO Gas (ppm)", obs.get("gas_co_ppm", 0), 35, 70),
        ("H2S Gas (ppm)", obs.get("gas_h2s_ppm", 0), 5, 20),
        ("Methane (%)", obs.get("methane_pct", 0), 0.5, 1.5),
        ("Roof Stability (0-1)", obs.get("roof_stability", 0), 0.7, 0.4, True),
        ("Earthquake Risk (0-1)", obs.get("earthquake_risk", 0), 0.2, 0.6),
    ]
    
    for sensor in sensors:
        name = sensor[0]
        value = sensor[1]
        
        if len(sensor) == 5:  # inverted threshold (roof stability)
            safe, danger, inverted = sensor[2], sensor[3], sensor[4]
            if value > safe:
                status = "✅ Safe"
            elif value < danger:
                status = "🔴 Danger"
            else:
                status = "⚠️ Caution"
        else:
            safe, danger = sensor[2], sensor[3]
            if value <= safe:
                status = "✅ Safe"
            elif value >= danger:
                status = "🔴 Danger"
            else:
                status = "⚠️ Caution"
        
        md += f"| {name} | {value:.2f} | {status} |\n"
    
    ventilation_status = "ON ✅" if obs.get("ventilation_on") else "OFF ❌"
    beams_status = "OK ✅" if obs.get("support_beams_ok") else "BAD ❌"
    near_miss = obs.get("last_near_miss_days", 0)
    
    md += f"| Ventilation | {ventilation_status} | {'✅' if obs.get('ventilation_on') else '🔴'} |\n"
    md += f"| Support Beams | {beams_status} | {'✅' if obs.get('support_beams_ok') else '🔴'} |\n"
    md += f"| Last Near Miss | {near_miss} days ago | {'⚠️' if near_miss < 7 else '✅'} |\n"
    
    return md

def reset_environment(difficulty: str):
    global current_obs, episode_data
    
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty}, timeout=10)
        resp.raise_for_status()
        current_obs = resp.json()
        
        episode_data = {
            "difficulty": difficulty,
            "reward": 0.0,
            "steps": 0,
            "accidents": 0,
            "decisions": []
        }
        
        sensors_md = format_sensors(current_obs)
        message = f"✅ Environment initialized with **{difficulty.upper()}** difficulty"
        
        return (message, sensors_md, float(0.0), int(0), int(0), "Ready")
    except Exception as e:
        return (f"❌ Error: {str(e)}", "Failed to initialize", float(0.0), int(0), int(0), "Error")

def take_action(decision: str, reasoning: str = ""):
    global current_obs, episode_data
    
    if current_obs is None:
        return ("❌ Please initialize environment first", "No data", float(0.0), int(0), int(0), "Not initialized")
    
    try:
        action_map = {"approve": 0, "postpone": 1, "scale_down": 2, "mandate_safety": 3}
        action_int = action_map.get(decision, 0)
        
        resp = requests.post(f"{ENV_URL}/step", json={"action": action_int}, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        
        current_obs = result.get("observation", current_obs)
        
        episode_data["reward"] = result.get("total_reward", 0.0)
        episode_data["steps"] = result.get("step_count", 0)
        episode_data["accidents"] = result.get("accidents", 0)
        
        message = f"**Decision: {decision.upper()}**\n\n{result.get('message', 'Action executed')}"
        if result.get("accident_occurred"):
            message += "\n\n🚨 **ACCIDENT OCCURRED!**"
        if result.get("done"):
            message += "\n\n🏁 **Episode Complete!**"
        
        return (message, format_sensors(current_obs), float(result.get("total_reward", 0.0)), 
                int(result.get("step_count", 0)), int(result.get("accidents", 0)), 
                "Complete" if result.get("done") else "In Progress")
    except Exception as e:
        return (f"❌ Error: {str(e)}", format_sensors(current_obs), float(episode_data["reward"]), 
                int(episode_data["steps"]), int(episode_data["accidents"]), "Error")

def run_baseline_agent(difficulty: str):
    global current_obs, episode_data
    
    try:
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty}, timeout=10)
        reset_resp.raise_for_status()
        current_obs = reset_resp.json()
        
        episode_data = {"difficulty": difficulty, "reward": 0.0, "steps": 0, "accidents": 0, "decisions": []}
        
        risk_factors = []
        if current_obs.get("gas_co_ppm", 0) > 50: risk_factors.append("high CO")
        if current_obs.get("gas_h2s_ppm", 0) > 15: risk_factors.append("high H2S")
        if current_obs.get("methane_pct", 0) > 1.2: risk_factors.append("high methane")
        if current_obs.get("roof_stability", 1) < 0.5: risk_factors.append("unstable roof")
        if current_obs.get("earthquake_risk", 0) > 0.5: risk_factors.append("high quake risk")
        
        if len(risk_factors) >= 3:
            decision = "postpone"
            reasoning = f"Multiple risks detected. Postponing for safety."
        elif len(risk_factors) >= 1:
            decision = "mandate_safety"
            reasoning = f"Risks detected. Adding safety measures."
        else:
            decision = "approve"
            reasoning = "All conditions within safe limits."
        
        action_map = {"approve": 0, "postpone": 1, "scale_down": 2, "mandate_safety": 3}
        step_resp = requests.post(f"{ENV_URL}/step", json={"action": action_map[decision]}, timeout=10)
        step_resp.raise_for_status()
        result = step_resp.json()
        current_obs = result.get("observation", current_obs)
        
        return (f"✅ **Baseline Agent Decision: {decision.upper()}**\n\n**Reasoning:** {reasoning}\n\n**Outcome:** {result.get('message')}\n\n**Reward:** {result.get('reward', 0):.1f}\n**Total Reward:** {result.get('total_reward', 0):.1f}",
                f"✅ Environment initialized with **{difficulty.upper()}** difficulty",
                format_sensors(current_obs), float(result.get("total_reward", 0.0)),
                int(result.get("step_count", 0)), int(result.get("accidents", 0)), "In Progress")
    except Exception as e:
        error_msg = f"❌ Baseline agent error: {str(e)}"
        return (error_msg, error_msg, "Error", float(0.0), int(0), int(0), "Error")

# Gradio UI
with gr.Blocks(title="SafeDig RL Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⛏️ SafeDig: Mining Safety RL Environment")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Controls")
            difficulty = gr.Radio(["easy", "medium", "hard"], value="easy", label="Difficulty Level")
            init_btn = gr.Button("🔄 Initialize Environment", variant="primary", size="lg")
            
            gr.Markdown("### 🤖 Agent Decision")
            decision = gr.Radio(["approve", "postpone", "scale_down", "mandate_safety"], label="Choose Action")
            reasoning = gr.Textbox(label="Reasoning", placeholder="Optional: Explain the decision...", lines=2)
            action_btn = gr.Button("▶️ Execute Action", variant="primary", size="lg")
            result_display = gr.Markdown("*Result will appear here*")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📈 Live Sensor Data")
            sensor_display = gr.Markdown("*Initialize environment to see sensor readings*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Episode Metrics")
            reward_display = gr.Number(label="Cumulative Reward", value=0.0, interactive=False)
            steps_display = gr.Number(label="Steps Taken", value=0, interactive=False)
            accidents_display = gr.Number(label="Accidents", value=0, interactive=False)
            status_display = gr.Textbox(label="Status", value="Not initialized", interactive=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Run Baseline Agent")
            baseline_difficulty = gr.Radio(["easy", "medium", "hard"], value="easy", label="Baseline Difficulty")
            baseline_btn = gr.Button("🚀 Run Baseline Agent", variant="secondary")
            baseline_result = gr.Markdown()
    
    init_btn.click(reset_environment, inputs=difficulty, outputs=[result_display, sensor_display, reward_display, steps_display, accidents_display, status_display])
    action_btn.click(take_action, inputs=[decision, reasoning], outputs=[result_display, sensor_display, reward_display, steps_display, accidents_display, status_display])
    baseline_btn.click(run_baseline_agent, inputs=baseline_difficulty, outputs=[baseline_result, result_display, sensor_display, reward_display, steps_display, accidents_display, status_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)