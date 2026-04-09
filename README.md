---
title: SafeDig Env
emoji: ⛏️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# SafeDig Agent — RL-based Hazardous Task Safety Checker

An OpenEnv environment where an AI agent decides whether it is safe
to perform a dangerous digging/mining task based on sensor readings.

## API Endpoints

- `POST /reset` — Start a new episode
- `POST /step` — Take an action
- `GET /state` — Get current state
- `GET /health` — Health check
- `GET /docs` — Interactive API docs

## Action Space

- `approve` — Approve the digging task
- `postpone` — Postpone due to danger
- `scale_down` — Reduce scope of task
- `mandate_safety` — Add extra safety measures

## Observation Space

- gas_co_ppm, gas_h2s_ppm, methane_pct
- roof_stability, earthquake_risk
- ventilation_on, support_beams_ok
- last_near_miss_days