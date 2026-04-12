---
title: SafeDig Env
emoji: ⛏️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
app_file: server/app.py
pinned: false
---

# SafeDig: Multi-Task RL Environment for Mining Safety

## Overview

SafeDig is a comprehensive OpenEnv-style environment featuring three critical mining safety tasks where AI agents make high-stakes decisions under uncertainty.

**Use Cases:**
- Reinforcement Learning (RL)
- LLM reasoning evaluation
- Safety-critical AI research

## Tasks

- **⛏️ Safety Decision** — Approve/postpone/scale operations based on environmental hazards
- **🔍 Sensor Reliability** — Detect faulty vs real readings and decide on action
- **🚨 Rescue Coordination** — Emergency extraction decisions with resource constraints

## Quick Installation

```bash
git clone https://github.com/YOUR_USERNAME/safedig-env.git
cd safedig-env

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt