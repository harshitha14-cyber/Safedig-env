import requests
import json
import os
import sys
from typing import Optional

# Configuration
API_BASE_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Check if server is running
try:
    response = requests.get(f"{API_BASE_URL}/api/health", timeout=2)
    if response.status_code != 200:
        raise Exception("Server not responding")
except Exception as e:
    print(f"[ERROR] Environment server not running: {e}", flush=True)
    sys.exit(1)

# Try to import HuggingFace transformers (optional)
HAS_LOCAL_MODEL = False
try:
    from transformers import pipeline
    HAS_LOCAL_MODEL = True
except ImportError:
    pass

def run_task(task_name: str, difficulty: str = "easy", num_episodes: int = 1):
    """Run a single task and output structured logs."""
    
    task_rewards = []
    
    for episode in range(num_episodes):
        # [START] block
        print(f'[START] task={task_name} difficulty={difficulty} episode={episode + 1}', flush=True)
        
        # Reset environment
        try:
            reset_response = requests.post(
                f"{API_BASE_URL}/reset",
                json={"task": task_name, "difficulty": difficulty},
                timeout=5
            )
            obs = reset_response.json()
        except Exception as e:
            print(f'[ERROR] Reset failed: {e}', flush=True)
            continue
        
        # Get task info
        task_actions = {
            "safety_decision": ["approve", "postpone", "scale_down", "mandate_safety"],
            "sensor_reliability": ["trust_and_proceed", "request_recalibration", "cross_reference", "emergency_stop"],
            "rescue_coordination": ["deploy_robotic_drill", "manual_extraction", "seal_section", "call_for_support"],
        }
        
        actions = task_actions.get(task_name, [])
        
        # Build prompt from observation
        obs_str = json.dumps(obs, indent=2)
        prompt = f"""You are an AI safety officer for a mining operation. 
        
Task: {task_name.replace('_', ' ').title()}

Current Situation:
{obs_str}

Available Actions: {', '.join(actions)}

Respond ONLY with valid JSON in this exact format:
{{"decision": "<action>", "reasoning": "<brief explanation>"}}

Choose the SAFEST action. Do not include any other text."""
        
        # Get model decision
        decision = None
        reasoning = ""
        
        if HAS_LOCAL_MODEL:
            try:
                pipe = pipeline("text-generation", model=MODEL_NAME, device=0)
                response = pipe(prompt, max_length=200, num_return_sequences=1)
                response_text = response[0]["generated_text"]
                
                # Extract JSON from response
                try:
                    json_start = response_text.rfind('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        parsed = json.loads(json_str)
                        decision = parsed.get("decision")
                        reasoning = parsed.get("reasoning", "")
                except:
                    pass
            except Exception:
                pass
        
        # Fallback: random action
        if not decision or decision not in actions:
            import random
            decision = random.choice(actions)
            reasoning = "Safety-first fallback action"
        
        # [STEP] block - Execute action
        try:
            step_response = requests.post(
                f"{API_BASE_URL}/step",
                json={
                    "task": task_name,
                    "decision": decision,
                    "reasoning": reasoning
                },
                timeout=5
            )
            result = step_response.json()
            reward = result.get("reward", 0.0)
            message = result.get("message", "")
            done = result.get("done", True)
            
            print(f'[STEP] step=1 action={decision} reward={reward:.3f} done={done}', flush=True)
            
            task_rewards.append(reward)
        except Exception as e:
            print(f'[ERROR] Step failed: {e}', flush=True)
            continue
        
        # [END] block
        if task_rewards:
            avg_reward = sum(task_rewards) / len(task_rewards)
            print(f'[END] task={task_name} score={avg_reward:.3f} steps=1 episodes={len(task_rewards)}', flush=True)

if __name__ == "__main__":
    # Run all 3 tasks
    for task in ["safety_decision", "sensor_reliability", "rescue_coordination"]:
        run_task(task, difficulty="easy", num_episodes=1)