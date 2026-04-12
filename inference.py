import json
import os
import requests
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a mining safety officer AI.
Given sensor readings for a dig site, output ONLY valid JSON:
{"decision": "<approve|postpone|scale_down|mandate_safety>", "reasoning": "<one sentence>"}
Base your decision strictly on the sensor values."""


def run_episode(difficulty: str) -> Dict[str, Any]:
    """Run a single episode of the SafeDig environment."""
    
    # Reset environment
    try:
        reset_response = requests.post(
            f"{ENV_URL}/api/reset",
            json={"difficulty": difficulty},
            timeout=10
        )
        reset_response.raise_for_status()
        obs = reset_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error resetting environment: {e}")
        return {}

    print(json.dumps({
        "type": "[START]",
        "difficulty": difficulty,
        "obs": obs
    }))

    # Build prompt from observation
    user_msg = f"""Site sensor readings:
- CO gas: {obs.get('gas_co_ppm', 0)} ppm (danger > 70)
- H2S gas: {obs.get('gas_h2s_ppm', 0)} ppm (danger > 20)
- Methane: {obs.get('methane_pct', 0)}% (danger > 1.5)
- Roof stability: {obs.get('roof_stability', 0)} (danger < 0.4)
- Earthquake risk: {obs.get('earthquake_risk', 0)} (danger > 0.6)
- Ventilation on: {obs.get('ventilation_on', False)}
- Support beams OK: {obs.get('support_beams_ok', False)}
- Last near miss: {obs.get('last_near_miss_days', 0)} days ago
Should the digging task be approved today?"""

    # Get LLM decision
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=100
        )

        # Extract and clean response
        raw = response.choices[0].message.content
        if raw is None:
            raw = ""
        raw = raw.strip()

    except Exception as e:
        print(f"Error getting LLM response: {e}")
        raw = ""

    # Parse JSON response
    try:
        parsed = json.loads(raw)
        # Validate required fields
        if "decision" not in parsed:
            parsed["decision"] = "postpone"
        if "reasoning" not in parsed:
            parsed["reasoning"] = "fallback decision"
    except json.JSONDecodeError:
        print(f"Warning: Could not parse LLM response: {raw}")
        parsed = {
            "decision": "postpone",
            "reasoning": "parse error fallback"
        }

    # Step environment
    try:
        step_response = requests.post(
            f"{ENV_URL}/api/step",
            json=parsed,
            timeout=10
        )
        step_response.raise_for_status()
        result = step_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error stepping environment: {e}")
        return {}

    # Get state
    try:
        state_response = requests.get(f"{ENV_URL}/api/state", timeout=10)
        state_response.raise_for_status()
        state = state_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting state: {e}")
        state = {}

    # Log step and end
    print(json.dumps({
        "type": "[STEP]",
        "action": parsed,
        "result": result
    }))

    print(json.dumps({
        "type": "[END]",
        "reward": result.get("reward", 0.0),
        "accident": result.get("accident_occurred", False),
        "state": state
    }))

    return result


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SafeDig RL Environment - Baseline Agent")
    print("="*60 + "\n")

    total_reward = 0.0
    num_episodes = 0

    for difficulty in ["easy", "medium", "hard"]:
        try:
            print(f"\n🎯 Running {difficulty.upper()} difficulty...")
            result = run_episode(difficulty)
            if result:
                reward = result.get("reward", 0.0)
                total_reward += reward
                num_episodes += 1
                print(f"✅ Reward: {reward:.3f}")
        except Exception as e:
            print(f"❌ Error in {difficulty} episode: {e}")

    print("\n" + "="*60)
    if num_episodes > 0:
        avg_reward = total_reward / num_episodes
        print(f"Total Episodes: {num_episodes}")
        print(f"Total Reward: {total_reward:.3f}")
        print(f"Average Reward: {avg_reward:.3f}")
    else:
        print("❌ No episodes completed successfully")
    print("="*60 + "\n")