from typing import Literal, Optional
from pydantic import BaseModel, Field

# ── TASK 1: Safety Decision ──
class SafeDigAction(BaseModel):
    decision: Literal["approve", "postpone", "scale_down", "mandate_safety"]
    reasoning: str = ""

class SafeDigObservation(BaseModel):
    gas_co_ppm: float
    gas_h2s_ppm: float
    methane_pct: float
    roof_stability: float
    earthquake_risk: float
    ventilation_on: bool
    support_beams_ok: bool
    last_near_miss_days: int
    reward: float
    done: bool
    accident_occurred: bool
    message: str

class SafeDigState(BaseModel):
    episode_id: str = ""
    task_difficulty: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    accidents: int = 0

# ── TASK 2: Sensor Reliability ──
class SensorReliabilityAction(BaseModel):
    decision: Literal["trust_and_proceed", "request_recalibration", "cross_reference", "emergency_stop"]
    reasoning: str = ""

class SensorReliabilityObservation(BaseModel):
    primary_co_ppm: float
    secondary_co_ppm: float
    primary_methane_pct: float
    secondary_methane_pct: float
    sensor_age_days: int
    conflict_detected: bool
    confidence_score: float
    ventilation_on: bool
    reward: float
    done: bool
    message: str

class SensorReliabilityState(BaseModel):
    episode_id: str = ""
    task_difficulty: str = ""
    step_count: int = 0
    total_reward: float = 0.0

# ── TASK 3: Rescue Coordination ──
class RescueAction(BaseModel):
    decision: Literal["deploy_robotic_drill", "manual_extraction", "seal_section", "call_for_support"]
    reasoning: str = ""

class RescueObservation(BaseModel):
    trapped_personnel: int
    oxygen_remaining_pct: float
    path_obstruction: float
    structural_risk: float
    available_resources: int
    time_elapsed_minutes: int
    reward: float
    done: bool
    casualties: int
    message: str

class RescueState(BaseModel):
    episode_id: str = ""
    task_difficulty: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    casualties: int = 0