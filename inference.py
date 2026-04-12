"""
Inference Script for SafeDig RL Environment
===================================
"""

import json
import os
import requests
from typing import List, Optional, Dict, Any

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

TASK_NAME = os.getenv("TASK_NAME", "safedig")
BENCHMARK = os.getenv("BENCHMARK", "openenv")


def log_start(task: str, env: str, model: str) -> None:
    """Print START block in required format"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Print STEP block in required format"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Print END block in required format"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_episode(difficulty: str, episode_num: int) -> tuple:
    """Run a single episode of the SafeDig environment."""
    
    rewards = []
    steps_taken = 0
    error = None
    done = False
    
    # Reset environment
    try:
        reset_response = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": difficulty},
            timeout=10
        )
        reset_response.raise_for_status()
        obs = reset_response.json()
    except requests.exceptions.RequestException as e:
        error = str(e)
        return rewards, steps_taken, error, done

    # Build prompt from observation
    user_msg = f"""Site sensor readings:
- CO gas: {obs.get('gas_co_ppm', 0)} ppm (danger > 70, caution > 35)
- H2S gas: {obs.get('gas_h2s_ppm', 0)} ppm (danger > 20, caution > 5)
- Methane: {obs.get('methane_pct', 0)}% (danger > 1.5, caution > 0.5)
- Roof stability: {obs.get('roof_stability', 0)} (danger < 0.4, caution < 0.7)
- Earthquake risk: {obs.get('earthquake_risk', 0)} (danger > 0.6, caution > 0.2)
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

        raw = response.choices[0].message.content
        if raw is None:
            raw = ""
        raw = raw.strip()

    except Exception as e:
        error = str(e)
        raw = ""

    # Parse JSON response
    try:
        parsed = json.loads(raw)
        if "decision" not in parsed:
            parsed["decision"] = "postpone"
        if "reasoning" not in parsed:
            parsed["reasoning"] = "fallback decision"
    except json.JSONDecodeError:
        parsed = {
            "decision": "postpone",
            "reasoning": "parse error fallback"
        }

    # Step environment
    try:
        step_response = requests.post(
            f"{ENV_URL}/step",
            json=parsed,
            timeout=10
        )
        step_response.raise_for_status()
        result = step_response.json()
        
        reward = result.get("reward", 0.0)
        done = result.get("done", True)
        rewards.append(reward)
        steps_taken = 1
        
        # Log step in required format
        log_step(step=1, action=parsed["decision"], reward=reward, done=done, error=error)
        
    except requests.exceptions.RequestException as e:
        error = str(e)
        log_step(step=1, action="error", reward=0.0, done=True, error=error)

    return rewards, steps_taken, error, done


def main() -> None:
    """Main inference function"""
    
    # Log start
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    all_rewards: List[float] = []
    total_steps = 0
    success = False
    
    try:
        # Run episodes for each difficulty
        for idx, difficulty in enumerate(["easy", "medium", "hard"], 1):
            rewards, steps, error, done = run_episode(difficulty, idx)
            all_rewards.extend(rewards)
            total_steps += steps
            
            if error:
                print(f"[DEBUG] Error in {difficulty}: {error}", flush=True)
        
        # Calculate final score (average of all rewards)
        if all_rewards:
            score = sum(all_rewards) / len(all_rewards)
            score = min(max(score, 0.0), 1.0)
            success = score >= 0.5
        else:
            score = 0.0
            success = False
            
    except Exception as e:
        print(f"[DEBUG] Main error: {e}", flush=True)
        score = 0.0
        success = False
        all_rewards = []
        total_steps = 0
    
    finally:
        # Always log END
        log_end(success=success, steps=total_steps, score=score, rewards=all_rewards)


if __name__ == "__main__":
    main()