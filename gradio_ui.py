"""
SafeDig RL Environment - Gradio Web UI
Interactive dashboard for mining safety decision-making
Run: python gradio_ui.py
"""

import os
import json
import requests
import gradio as gr

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    exit(1)

# Configuration
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Global state
current_obs = None
current_state = None
episode_data = {
    "difficulty": "easy",
    "reward": 0.0,
    "steps": 0,
    "accidents": 0,
    "decisions": []
}


def reset_environment(difficulty: str):
    """Reset the environment and return observation"""
    global current_obs, current_state, episode_data
    
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty}, timeout=10)
        resp.raise_for_status()
        current_obs = resp.json()
        
        # Get state
        state_resp = requests.get(f"{ENV_URL}/state", timeout=10)
        state_resp.raise_for_status()
        current_state = state_resp.json()
        
        episode_data = {
            "difficulty": difficulty,
            "reward": 0.0,
            "steps": 0,
            "accidents": 0,
            "decisions": []
        }
        
        # Format sensor data
        sensors_md = format_sensors(current_obs)
        message = f"✅ Environment initialized with **{difficulty.upper()}** difficulty\n\n{current_obs.get('message', 'Ready')}"
        
        return (
            message,
            sensors_md,
            float(0.0),
            int(0),
            int(0),
            "Ready",
        )
    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            "Failed to initialize",
            float(0.0),
            int(0),
            int(0),
            "Error",
        )


def format_sensors(obs):
    """Format sensor data as markdown table"""
    if not obs:
        return "No data"
    
    thresholds = {
        "gas_co_ppm": (35, 70),
        "gas_h2s_ppm": (5, 20),
        "methane_pct": (0.5, 1.5),
        "roof_stability": (0.7, 0.4, True),
        "earthquake_risk": (0.2, 0.6),
    }
    
    def get_status(key, value):
        if key not in thresholds:
            return "ℹ️"
        
        if key == "roof_stability":
            safe, danger, inverted = thresholds[key]
            if value > safe:
                return "✅ Safe"
            elif value < danger:
                return "🔴 Danger"
            else:
                return "⚠️ Caution"
        else:
            safe, danger = thresholds[key][:2]
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
        status = get_status(name.split("(")[0].strip().lower().replace(" ", "_"), value)
        if isinstance(value, float):
            md += f"| {name} | {value:.2f} | {status} |\n"
        else:
            md += f"| {name} | {value} | {status} |\n"
    
    # Equipment status
    md += f"| Ventilation | {'ON ✅' if obs.get('ventilation_on') else 'OFF ❌'} | {'✅' if obs.get('ventilation_on') else '⚠️'} |\n"
    md += f"| Support Beams | {'OK ✅' if obs.get('support_beams_ok') else 'BAD ❌'} | {'✅' if obs.get('support_beams_ok') else '🔴'} |\n"
    md += f"| Last Near Miss | {obs.get('last_near_miss_days', 0)} days ago | {'⚠️' if obs.get('last_near_miss_days', 0) < 7 else '✅'} |\n"
    
    return md


def take_action(decision, reasoning=""):
    """Take an action in the environment"""
    global current_obs, current_state, episode_data
    
    if current_obs is None:
        return (
            "❌ Please initialize environment first",
            "No data",
            float(0.0),
            int(0),
            int(0),
            "Not initialized"
        )
    
    try:
        # Send action
        action_data = {
            "decision": decision,
            "reasoning": reasoning or f"Agent decided to {decision}"
        }
        
        resp = requests.post(f"{ENV_URL}/step", json=action_data, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        
        # Get updated state
        state_resp = requests.get(f"{ENV_URL}/state", timeout=10)
        state_resp.raise_for_status()
        current_state = state_resp.json()
        
        # Update episode data
        episode_data["reward"] = result.get("reward", 0.0)
        episode_data["steps"] = current_state.get("step_count", 0)
        episode_data["accidents"] = current_state.get("accidents", 0)
        episode_data["decisions"].append({
            "action": decision,
            "reward": result.get("reward", 0.0),
            "message": result.get("message", "")
        })
        
        # Format message
        message = f"**Decision: {decision.upper()}**\n\n{result.get('message', 'Action executed')}"
        
        if result.get("accident_occurred"):
            message += "\n\n🚨 **ACCIDENT OCCURRED!**"
        
        reward = float(result.get("reward", 0.0))
        
        return (
            message,
            format_sensors(current_obs),
            reward,
            int(current_state.get("step_count", 0)),
            int(current_state.get("accidents", 0)),
            "Complete" if result.get("done") else "In Progress"
        )
    
    except Exception as e:
        return (
            f"❌ Error taking action: {str(e)}",
            format_sensors(current_obs),
            float(0.0),
            int(episode_data["steps"]),
            int(episode_data["accidents"]),
            "Error"
        )


def run_baseline_agent(difficulty):
    """Run a baseline agent for demonstration"""
    global current_obs
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        
        # Reset
        resp = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty}, timeout=10)
        resp.raise_for_status()
        current_obs = resp.json()
        
        # Build prompt
        prompt = f"""You are a mining safety officer. Given these sensor readings, decide on action.
        
Sensor readings:
- CO Gas: {current_obs.get('gas_co_ppm', 0)} ppm (danger > 70)
- H2S Gas: {current_obs.get('gas_h2s_ppm', 0)} ppm (danger > 20)
- Methane: {current_obs.get('methane_pct', 0)}% (danger > 1.5)
- Roof stability: {current_obs.get('roof_stability', 0)} (danger < 0.4)
- Earthquake risk: {current_obs.get('earthquake_risk', 0)} (danger > 0.6)
- Ventilation: {current_obs.get('ventilation_on', False)}
- Support beams: {current_obs.get('support_beams_ok', False)}

Output ONLY valid JSON with decision and reasoning:
{{"decision": "approve|postpone|scale_down|mandate_safety", "reasoning": "<brief reason>"}}"""
        
        # Get LLM decision
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        raw = response.choices[0].message.content
        if raw is None:
            raw = ""
        raw = raw.strip()
        
        parsed = json.loads(raw)
        
        # Execute
        step_resp = requests.post(f"{ENV_URL}/step", json=parsed, timeout=10)
        step_resp.raise_for_status()
        result = step_resp.json()
        
        return f"✅ **{parsed['decision'].upper()}**: {result.get('message', 'Action taken')}\nReason: {parsed['reasoning']}"
    
    except Exception as e:
        return f"❌ Baseline agent error: {str(e)}"


# Build Gradio interface
with gr.Blocks(
    title="SafeDig RL Environment",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("# ⛏️ SafeDig: Mining Safety RL Environment")
    gr.Markdown("Real-time decision-making for hazardous mining operations. Help the agent learn to make safe choices.")
    
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
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Episode Metrics")
            reward_display = gr.Number(label="Cumulative Reward", value=0.0, interactive=False)
            steps_display = gr.Number(label="Steps Taken", value=0, interactive=False)
            accidents_display = gr.Number(label="Accidents", value=0, interactive=False)
            status_display = gr.Textbox(label="Status", value="Not initialized", interactive=False)
    
    gr.Markdown("### 📈 Live Sensor Data")
    sensor_display = gr.Markdown("Initialize environment to see sensor readings")
    
    gr.Markdown("### 🤖 Agent Decision")
    with gr.Row():
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
    result_display = gr.Markdown("Result will appear here")
    
    gr.Markdown("### 🤖 Run Baseline Agent")
    baseline_difficulty = gr.Radio(
        ["easy", "medium", "hard"],
        value="easy",
        label="Baseline Difficulty",
        info="Let the LLM agent make decisions"
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
        outputs=baseline_result
    )
    
    gr.Markdown("""
    ---
    ### 📚 How It Works
    
    **Actions & Rewards:**
    - **Approve** (+10 if safe, -50 if danger): Approve the task
    - **Postpone** (+15 if danger, -5 if safe): Delay the task
    - **Scale Down** (+8 if danger, +5 if safe): Reduce scope
    - **Mandate Safety** (+12 if danger, +3 if safe): Add measures
    
    **Difficulty Levels:**
    - **Easy**: Clearly safe or clearly dangerous
    - **Medium**: Mixed conditions requiring analysis
    - **Hard**: Borderline sensor values
    
    ---
    🔗 [API Documentation](/docs) | 📖 [GitHub](https://github.com/spiderweb2006/openenv-project) | 🏆 [OpenEnv Framework](https://github.com/openenvhq/openenv)
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )