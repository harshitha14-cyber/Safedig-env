from pydantic import BaseModel
from typing import Literal

class SafeDigAction(BaseModel):
    decision: Literal["approve", "postpone", "scale_down", "mandate_safety"]
    reasoning: str = ""  # LLM agent explains why

class SafeDigObservation(BaseModel):
    gas_co_ppm: float
    gas_h2s_ppm: float
    methane_pct: float
    roof_stability: float     # 0.0 (bad) to 1.0 (perfect)
    earthquake_risk: float    # 0.0 to 1.0
    ventilation_on: bool
    support_beams_ok: bool
    last_near_miss_days: int
    reward: float
    done: bool
    accident_occurred: bool
    message: str

class SafeDigState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    accidents: int = 0
    task_difficulty: str = "easy"   # easy / medium / hard