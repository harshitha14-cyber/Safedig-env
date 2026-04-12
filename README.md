---
title: SafeDig Env
emoji: ⛏️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
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
curl -X POST http://localhost:7860/api/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty":"easy"}'

# Take an action
curl -X POST http://localhost:7860/api/step \
  -H "Content-Type: application/json" \
  -d '{"decision":"postpone","reasoning":"gas levels too high"}'

# Check current state
curl http://localhost:7860/api/state
```

## 🏗️ Action Space

The agent can take one of 4 actions:

| Action | Safe Zone | Caution Zone | Danger Zone |
|--------|-----------|--------------|-------------|
| `approve` | +10.0 (Perfect) | -10.0 (Warning!) | -50.0 (ACCIDENT) |
| `postpone` | -5.0 (Too cautious) | +8.0 (Smart) | +15.0 (Excellent) |
| `scale_down` | -2.0 (Unnecessary) | +5.0 (Acceptable) | +8.0 (Partial credit) |
| `mandate_safety` | -3.0 (Wasteful) | +6.0 (Good precaution) | +12.0 (Excellent) |

**Normalized scores (0-1 scale):**
- 1.0 = Perfect decision
- 0.8-0.99 = Good decision
- 0.6-0.79 = Acceptable
- 0.4-0.59 = Poor
- <0.4 = Dangerous

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

BASE_URL = "http://localhost:7860"
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
├── server/
│   ├── app.py          ← FastAPI server (moved)
│   ├── environment.py  ← RL logic (moved)
│   ├── models.py       ← Pydantic models (moved)
│   └── requirements.txt
├── inference.py        ← Baseline agent
├── Dockerfile
├── requirements.txt    ← Root requirements
└── README.md
```

## 🔗 API Endpoints

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/api/reset` | POST | `{"difficulty": "easy|medium|hard"}` | `Observation` (JSON) |
| `/api/step` | POST | `{"decision": "...", "reasoning": "..."}` | `Observation` (JSON) |
| `/api/state` | GET | — | `State` (JSON) |
| `/api/health` | GET | — | `{"status": "ok"}` |
| `/api/docs` | GET | — | Interactive Swagger UI |

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
export ENV_URL="http://localhost:7860"                # Environment API URL
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