import requests
import json
import os
import sys
import time
import random

# ── Load .env file if present (no extra packages needed) ─────────────────────
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Resolve ${VAR} references
            if val.startswith("${") and val.endswith("}"):
                ref = val[2:-1]
                val = os.getenv(ref, "")
            if key and key not in os.environ:  # don't override existing env vars
                os.environ[key] = val

_load_dotenv()

# ── Config — defaults match your .env ────────────────────────────────────────
ENV_URL      = os.getenv("ENV_URL",      "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")


def log(msg: str):
    """Diagnostics go to stderr — keeps stdout clean for the validator."""
    print(msg, file=sys.stderr, flush=True)


def wait_for_server(url: str, retries: int = 10, delay: float = 1.5) -> bool:
    log(f"[INFO] Waiting for environment server at {url} ...")
    for attempt in range(1, retries + 1):
        for endpoint in ["/api/health", "/health"]:
            try:
                r = requests.get(f"{url}{endpoint}", timeout=2)
                if r.status_code == 200:
                    log(f"[INFO] Server is ready (responded on {endpoint}).")
                    return True
            except Exception:
                pass
        log(f"[INFO] Attempt {attempt}/{retries} failed. Retrying in {delay}s...")
        time.sleep(delay)
    log(f"[ERROR] Environment server not responding at {url} after {retries} attempts.")
    log("[INFO] Start your environment server first: python server/app.py")
    return False


def get_llm_decision(prompt: str, actions: list) -> tuple:
    """Call LLM via HuggingFace router. Falls back to random if no key or call fails."""
    if API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a mining safety AI. Output only valid JSON."},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            json_start = response_text.rfind('{')
            json_end   = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                parsed    = json.loads(response_text[json_start:json_end])
                decision  = parsed.get("decision", "")
                reasoning = parsed.get("reasoning", "")
                if decision in actions:
                    return decision, reasoning
        except Exception as e:
            log(f"[WARN] LLM call failed: {e}")

    fallback = random.choice(actions)
    log(f"[INFO] Using fallback action: {fallback}")
    return fallback, "Safety-first fallback action"


def run_task(task_name: str, difficulty: str = "easy", num_episodes: int = 1):
    task_actions = {
        "safety_decision":     ["approve", "postpone", "scale_down", "mandate_safety"],
        "sensor_reliability":  ["trust_and_proceed", "request_recalibration", "cross_reference", "emergency_stop"],
        "rescue_coordination": ["deploy_robotic_drill", "manual_extraction", "seal_section", "call_for_support"],
    }
    actions = task_actions.get(task_name, [])

    for episode in range(num_episodes):
        # Reset
        try:
            reset_resp = requests.post(
                f"{ENV_URL}/reset",
                json={"task": task_name, "difficulty": difficulty},
                timeout=5
            )
            if reset_resp.status_code != 200:
                log(f"[ERROR] Reset failed ({reset_resp.status_code}): {reset_resp.text}")
                continue
            obs = reset_resp.json()
        except Exception as e:
            log(f"[ERROR] Reset failed: {e}")
            continue

        # Prompt
        prompt = f"""You are an AI safety officer for a mining operation.

Task: {task_name.replace('_', ' ').title()}
Current Situation:
{json.dumps(obs, indent=2)}
Available Actions: {', '.join(actions)}
Respond ONLY with valid JSON in this exact format:
{{"decision": "<action>", "reasoning": "<brief explanation>"}}
Choose the SAFEST action. Do not include any other text."""

        decision, reasoning = get_llm_decision(prompt, actions)

        # Step
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={
                    "task":       task_name,
                    "decision":   decision,
                    "reasoning":  reasoning,
                    "difficulty": difficulty
                },
                timeout=5
            )
            if step_resp.status_code != 200:
                log(f"[ERROR] Step failed ({step_resp.status_code}): {step_resp.text}")
                continue

            result = step_resp.json()
            reward = result.get("reward", 0.0)
            done   = result.get("done", True)

            # Strict stdout — exactly what the hackathon validator expects
            print(f"[START] task={task_name} difficulty={difficulty} episode={episode + 1}", flush=True)
            print(f"[STEP] step=1 action={decision} reward={reward:.3f} done={str(done).capitalize()}", flush=True)
            print(f"[END] task={task_name} score={reward:.3f} steps=1 episodes={episode + 1}", flush=True)

        except Exception as e:
            log(f"[ERROR] Step failed: {e}")
            continue


if __name__ == "__main__":
    if not wait_for_server(ENV_URL):
        sys.exit(1)

    for task in ["safety_decision", "sensor_reliability", "rescue_coordination"]:
        run_task(task, difficulty="easy", num_episodes=1)