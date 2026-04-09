---
title: SafeDig Env
emoji: ⛏️
colorFrom: yellow
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# SafeDig Agent — RL-based Hazardous Task Safety Checker

An OpenEnv environment where an AI agent decides whether it is safe to perform a dangerous digging/mining task based on sensor readings.

## 🎯 Problem Statement

Mining operations face critical safety challenges. Before approving a dig task, a safety officer must assess multiple environmental factors (gas levels, structural integrity, seismic activity) and make a decision: **approve**, **postpone**, **scale down**, or **mandate additional safety measures**.

This RL environment simulates that decision-making process, training agents to make safety-critical judgments.

## 📋 Quick Start

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app:app --reload

# In another terminal, test the environment
python main.py
```

### Test a Single Scenario
```bash
# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty":"easy"}'

# Take an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"decision":"postpone","reasoning":"gas levels too high"}'

# Check current state
curl http://localhost:8000/state
```

## 🏗️ Action Space

The agent can take one of 4 actions:

| Action | Description | Reward (if correct) | Reward (if wrong) |
|--------|-------------|-------------------|-------------------|
| `approve` | Approve digging task | +10.0 (safe → +1.0) | -50.0 (danger → 0.0) |
| `postpone` | Delay task due to danger | +15.0 (danger → 1.0) | -5.0 (safe → ~0.92) |
| `scale_down` | Reduce task scope | +8.0 (danger → 0.83) | +5.0 (safe → 0.92) |
| `mandate_safety` | Add safety measures | +12.0 (danger → 0.94) | +3.0 (safe → 0.88) |

## 👀 Observation Space

The agent observes 9 sensor readings + metadata:

### Gas Levels
- `gas_co_ppm` (float) — Carbon monoxide [Safe: <35 | Danger: >70]
- `gas_h2s_ppm` (float) — Hydrogen sulfide [Safe: <5 | Danger: >20]
- `methane_pct` (float) — Methane concentration [Safe: <0.5% | Danger: >1.5%]

### Structural & Environmental
- `roof_stability` (0.0-1.0) — Ceiling integrity [Safe: >0.7 | Danger: <0.4]
- `earthquake_risk` (0.0-1.0) — Seismic activity level [Safe: <0.2 | Danger: >0.6]

### Equipment Status
- `ventilation_on` (bool) — Is ventilation running?
- `support_beams_ok` (bool) — Are support structures sound?
- `last_near_miss_days` (int) — Days since last accident

### Episode Info
- `reward` (0.0-1.0) — Reward for last action
- `done` (bool) — Episode terminated?
- `accident_occurred` (bool) — Did an accident happen?
- `message` (str) — Human-readable feedback

## 📊 Task Difficulties

### Easy (Clear scenarios)
- Clearly safe conditions OR clearly dangerous
- Example safe: All readings normal, equipment working
- Example danger: Multiple hazards active simultaneously

### Medium (Mixed conditions)
- Some sensors safe, some borderline
- Requires careful analysis of trade-offs
- Realistic operational decision-making

### Hard (Borderline everything)
- All sensor values near thresholds
- Requires nuanced judgment
- Most challenging for agents

## 🤖 Training an Agent

Example inference script structure:
```python
import requests
import json
from openai import OpenAI

BASE_URL = "http://localhost:8000"
client = OpenAI(api_key="your_key", base_url="your_api_base")

# Reset environment
obs = requests.post(f"{BASE_URL}/reset", json={"difficulty": "easy"}).json()

# Build prompt from observation
prompt = f"Sensor readings: CO={obs['gas_co_ppm']}ppm, H2S={obs['gas_h2s_ppm']}ppm..."

# Get LLM decision
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100
)

# Parse decision and step
decision = json.loads(response.choices[0].message.content)
result = requests.post(f"{BASE_URL}/step", json=decision).json()

print(f"Reward: {result['reward']:.3f}")
```

## 📁 Project Structure

```
safedig-env/
├── models.py          ← Pydantic models (Action, Observation, State)
├── environment.py     ← RL logic (reset, step, rewards)
├── app.py             ← FastAPI server
├── inference.py       ← Baseline agent (LLM-based)
├── main.py            ← Local testing script
├── openenv.yaml       ← OpenEnv spec
├── pyproject.toml     ← Project metadata
├── requirements.txt   ← Dependencies
├── Dockerfile         ← Container config
└── README.md          ← This file
```

## 🔗 API Endpoints

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/reset` | POST | `{"difficulty": "easy\|medium\|hard"}` | `Observation` (JSON) |
| `/step` | POST | `{"decision": "...", "reasoning": "..."}` | `Observation` (JSON) |
| `/state` | GET | — | `State` (JSON) |
| `/health` | GET | — | `{"status": "ok"}` |
| `/docs` | GET | — | Interactive Swagger UI |

## 📈 Reward Normalization

Raw rewards are normalized to [0.0, 1.0]:
```
normalized_reward = (raw_reward + 50) / 65
```

This maps the range [-50, +15] to [0.0, 1.0], where:
- 0.0 = Catastrophic failure (missed critical danger)
- 0.5 = Neutral (minor mistakes)
- 1.0 = Perfect decision (correct action with bonus)

## 🧪 Environment Configuration

Pass these environment variables to `inference.py`:
```bash
export API_BASE_URL="https://api.openai.com/v1"      # LLM API endpoint
export MODEL_NAME="gpt-4o-mini"                       # Model identifier
export HF_TOKEN="hf_xxxxx"                            # Hugging Face token
export ENV_URL="http://localhost:8000"                # Environment API URL
```

## 🚀 Deployment

### Local Docker
```bash
docker build -t safedig .
docker run -p 7860:7860 safedig
```

### Hugging Face Spaces
1. Create Space: https://huggingface.co/new-space
2. Select "Docker" SDK
3. Push this repo
4. Space will auto-deploy

## 📝 Logging Format

`inference.py` must emit structured logs for automated evaluation:

```json
{"type": "[START]", "difficulty": "easy", "obs": {...}}
{"type": "[STEP]", "action": {...}, "result": {...}}
{"type": "[END]", "reward": 0.85, "accident": false, "state": {...}}
```

## 🏆 Performance Metrics

Expected agent performance across difficulties:
- **Easy**: 0.8-0.95 reward (clear scenarios)
- **Medium**: 0.6-0.8 reward (mixed conditions)
- **Hard**: 0.5-0.7 reward (borderline judgments)

## 📚 References

- [OpenEnv Framework](https://github.com/openenvhq/openenv)
- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- Mining Safety Standards (MSHA, ILO)

---

**Built with ❤️ for the Meta PyTorch Hackathon**