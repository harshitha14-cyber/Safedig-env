from dotenv import load_dotenv
load_dotenv()

import json, os, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a mining safety officer AI.
Given sensor readings for a dig site, output ONLY valid JSON:
{"decision": "<approve|postpone|scale_down|mandate_safety>", "reasoning": "<one sentence>"}
Base your decision strictly on the sensor values."""

def run_episode(difficulty: str) -> dict:
    # Reset environment
    obs = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty}).json()
    print(json.dumps({"type": "[START]", "difficulty": difficulty, "obs": obs}))

    # Build prompt from observation
    user_msg = f"""Site sensor readings:
- CO gas: {obs['gas_co_ppm']} ppm (danger > 70)
- H2S gas: {obs['gas_h2s_ppm']} ppm (danger > 20)
- Methane: {obs['methane_pct']}% (danger > 1.5)
- Roof stability: {obs['roof_stability']} (danger < 0.4)
- Earthquake risk: {obs['earthquake_risk']} (danger > 0.6)
- Ventilation on: {obs['ventilation_on']}
- Support beams OK: {obs['support_beams_ok']}
- Last near miss: {obs['last_near_miss_days']} days ago
Should the digging task be approved today?"""

    # LLM decides
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        max_tokens=100
    )
    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except:
        parsed = {"decision": "postpone", "reasoning": "parse error fallback"}

    # Step environment
    result = requests.post(f"{ENV_URL}/step", json=parsed).json()
    state  = requests.get(f"{ENV_URL}/state").json()

    print(json.dumps({"type": "[STEP]", "action": parsed, "result": result}))
    print(json.dumps({"type": "[END]",  "reward": result["reward"],
                      "accident": result["accident_occurred"], "state": state}))
    return result

if __name__ == "__main__":
    total = 0.0
    for diff in ["easy", "medium", "hard"]:
        r = run_episode(diff)
        total += r["reward"]
    print(f"\nFinal total reward: {total:.3f}")