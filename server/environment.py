import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import random
from models import (
    SafeDigAction, SafeDigObservation, SafeDigState,
    SensorReliabilityAction, SensorReliabilityObservation, SensorReliabilityState,
    RescueAction, RescueObservation, RescueState,
)

THRESHOLDS = {
    "gas_co_ppm":     {"safe": 35,  "danger": 70},
    "gas_h2s_ppm":    {"safe": 5,   "danger": 20},
    "methane_pct":    {"safe": 0.5, "danger": 1.5},
    "roof_stability": {"safe": 0.7, "danger": 0.4},
    "earthquake_risk":{"safe": 0.2, "danger": 0.6},
}

def normalize(reward: float, min_r: float, max_r: float) -> float:
    """
    Normalize reward to [0, 1].
    Worst possible action → 0.0, best possible action → 1.0.
    This fixes the bug where -50 reward was mapping to 0.0 score,
    making accidents look the same as 'no action taken'.
    """
    return max(0.0, min(1.0, (reward - min_r) / (max_r - min_r)))


# ═══════════════════════════════════════════════
# TASK 1 — Static Safety Decision
# ═══════════════════════════════════════════════
class SafeDigEnvironment:
    # reward range: -50 (approve dangerous) to +15 (postpone dangerous)
    _MIN_R, _MAX_R = -50.0, +15.0

    def __init__(self):
        self._state = SafeDigState()
        self._obs = None
        self._current_scenario = {}

    def _generate_scenario(self, difficulty="easy") -> dict:
        if difficulty == "easy":
            roll = random.random()
            if roll < 0.6:
                # 60% — clearly safe, low values well within limits
                return {
                    "gas_co_ppm": random.uniform(5, 25),
                    "gas_h2s_ppm": random.uniform(1, 4),
                    "methane_pct": random.uniform(0.05, 0.35),
                    "roof_stability": random.uniform(0.80, 0.98),
                    "earthquake_risk": random.uniform(0.01, 0.12),
                    "ventilation_on": True,
                    "support_beams_ok": True,
                    "last_near_miss_days": random.randint(20, 60)
                }
            elif roll < 0.8:
                # 20% — clearly dangerous, obvious red flags
                return {
                    "gas_co_ppm": random.uniform(80, 120),
                    "gas_h2s_ppm": random.uniform(22, 35),
                    "methane_pct": random.uniform(1.8, 2.5),
                    "roof_stability": random.uniform(0.10, 0.30),
                    "earthquake_risk": random.uniform(0.70, 0.90),
                    "ventilation_on": False,
                    "support_beams_ok": False,
                    "last_near_miss_days": random.randint(1, 3)
                }
            else:
                # 20% — mild caution zone (borderline but learnable)
                return {
                    "gas_co_ppm": random.uniform(38, 55),
                    "gas_h2s_ppm": random.uniform(6, 12),
                    "methane_pct": random.uniform(0.55, 0.9),
                    "roof_stability": random.uniform(0.50, 0.68),
                    "earthquake_risk": random.uniform(0.22, 0.40),
                    "ventilation_on": random.choice([True, False]),
                    "support_beams_ok": True,
                    "last_near_miss_days": random.randint(5, 15)
                }
        elif difficulty == "medium":
            # Clearly safe OR clearly dangerous — requires reasoning, not ambiguous
            safe = random.random() > 0.5
            if safe:
                return {
                    "gas_co_ppm": random.uniform(15, 34),
                    "gas_h2s_ppm": random.uniform(2, 4.9),
                    "methane_pct": random.uniform(0.2, 0.49),
                    "roof_stability": random.uniform(0.72, 0.90),
                    "earthquake_risk": random.uniform(0.05, 0.18),
                    "ventilation_on": True,
                    "support_beams_ok": True,
                    "last_near_miss_days": random.randint(8, 25)
                }
            else:
                return {
                    "gas_co_ppm": random.uniform(72, 95),
                    "gas_h2s_ppm": random.uniform(21, 30),
                    "methane_pct": random.uniform(1.6, 2.2),
                    "roof_stability": random.uniform(0.15, 0.38),
                    "earthquake_risk": random.uniform(0.62, 0.80),
                    "ventilation_on": random.choice([True, False]),
                    "support_beams_ok": random.choice([True, False]),
                    "last_near_miss_days": random.randint(1, 8)
                }
        else:  # hard — all values sit in the caution zone, genuinely ambiguous
            return {
                "gas_co_ppm": random.uniform(37, 68),
                "gas_h2s_ppm": random.uniform(5.5, 18),
                "methane_pct": random.uniform(0.52, 1.44),
                "roof_stability": random.uniform(0.42, 0.68),
                "earthquake_risk": random.uniform(0.22, 0.57),
                "ventilation_on": random.choice([True, False]),
                "support_beams_ok": random.choice([True, False]),
                "last_near_miss_days": random.randint(3, 12)
            }

    def _is_actually_dangerous(self, s: dict) -> bool:
        return (
            s["gas_co_ppm"] > THRESHOLDS["gas_co_ppm"]["danger"] or
            s["gas_h2s_ppm"] > THRESHOLDS["gas_h2s_ppm"]["danger"] or
            s["methane_pct"] > THRESHOLDS["methane_pct"]["danger"] or
            s["roof_stability"] < THRESHOLDS["roof_stability"]["danger"] or
            s["earthquake_risk"] > THRESHOLDS["earthquake_risk"]["danger"] or
            not s["support_beams_ok"]
        )

    def _is_caution_zone(self, s: dict) -> bool:
        return (
            (THRESHOLDS["gas_co_ppm"]["safe"] < s["gas_co_ppm"] <= THRESHOLDS["gas_co_ppm"]["danger"]) or
            (THRESHOLDS["gas_h2s_ppm"]["safe"] < s["gas_h2s_ppm"] <= THRESHOLDS["gas_h2s_ppm"]["danger"]) or
            (THRESHOLDS["methane_pct"]["safe"] < s["methane_pct"] <= THRESHOLDS["methane_pct"]["danger"]) or
            (THRESHOLDS["roof_stability"]["danger"] <= s["roof_stability"] < THRESHOLDS["roof_stability"]["safe"]) or
            (THRESHOLDS["earthquake_risk"]["safe"] < s["earthquake_risk"] <= THRESHOLDS["earthquake_risk"]["danger"]) or
            (not s["ventilation_on"] and s["gas_co_ppm"] > 20) or
            (not s["support_beams_ok"] and s["roof_stability"] < 0.6)
        )

    def reset(self, difficulty: str = "easy") -> SafeDigObservation:
        self._current_scenario = self._generate_scenario(difficulty)
        self._state = SafeDigState(episode_id=str(uuid.uuid4()), task_difficulty=difficulty)
        s = self._current_scenario
        return SafeDigObservation(
            gas_co_ppm=s["gas_co_ppm"], gas_h2s_ppm=s["gas_h2s_ppm"],
            methane_pct=s["methane_pct"], roof_stability=s["roof_stability"],
            earthquake_risk=s["earthquake_risk"], ventilation_on=s["ventilation_on"],
            support_beams_ok=s["support_beams_ok"], last_near_miss_days=s["last_near_miss_days"],
            reward=0.0, done=False, accident_occurred=False,
            message="New shift starting. Assess the site."
        )

    def step(self, action: SafeDigAction) -> SafeDigObservation:
        s = self._current_scenario
        dangerous = self._is_actually_dangerous(s)
        caution   = self._is_caution_zone(s)
        reward = 0.0
        accident = False
        msg = ""

        if action.decision == "approve":
            if dangerous:
                reward = -50.0; accident = True; self._state.accidents += 1
                msg = "💥 ACCIDENT! Conditions were dangerous but task was approved."
            elif caution:
                reward = -10.0
                msg = "⚠️ WARNING! Borderline conditions — should have postponed or added safety measures."
            else:
                reward = +10.0
                msg = "✅ Perfect! Conditions were safe. Task approved and completed successfully."

        elif action.decision == "postpone":
            if dangerous:
                reward = +15.0
                msg = "✅ Excellent! Conditions were dangerous. Postponing was the right decision."
            elif caution:
                reward = +8.0
                msg = "✅ Good call! Conditions are borderline — waiting is prudent."
            else:
                reward = -5.0
                msg = "⚠️ Over-cautious. Conditions were actually safe. Consider approving next time."

        elif action.decision == "scale_down":
            if dangerous:
                reward = +8.0
                msg = "⚠️ Acceptable: Conditions dangerous but scaling down reduces risk. Postponing would be better."
            elif caution:
                reward = +5.0
                msg = "✅ Reasonable compromise. Conditions borderline — scaling down is acceptable."
            else:
                reward = -2.0
                msg = "⚠️ Unnecessary reduction. Conditions were safe, no need to scale down."

        elif action.decision == "mandate_safety":
            if dangerous:
                reward = +12.0
                msg = "✅ Excellent! Mandating extra safety measures in dangerous conditions is very responsible."
            elif caution:
                reward = +6.0
                msg = "✅ Good precaution. Extra safety measures in caution zone is wise."
            else:
                reward = -3.0
                msg = "⚠️ Unnecessary expense. Conditions were completely safe."

        self._state.step_count += 1
        self._state.total_reward += reward
        score = normalize(reward, self._MIN_R, self._MAX_R)

        return SafeDigObservation(
            gas_co_ppm=s["gas_co_ppm"], gas_h2s_ppm=s["gas_h2s_ppm"],
            methane_pct=s["methane_pct"], roof_stability=s["roof_stability"],
            earthquake_risk=s["earthquake_risk"], ventilation_on=s["ventilation_on"],
            support_beams_ok=s["support_beams_ok"], last_near_miss_days=s["last_near_miss_days"],
            reward=score, done=True, accident_occurred=accident, message=msg
        )

    @property
    def state(self) -> SafeDigState:
        return self._state


# ═══════════════════════════════════════════════
# TASK 2 — Sensor Reliability & Ghost Hazards
# ═══════════════════════════════════════════════
class SensorReliabilityEnvironment:
    # reward range: -50 (trust faulty sensor in real danger) to +15 (cross_reference real danger)
    _MIN_R, _MAX_R = -50.0, +15.0

    def __init__(self):
        self._state = SensorReliabilityState()
        self._scenario = {}
        self._real_danger = False

    def _generate_scenario(self, difficulty="easy"):
        sensor_age = random.randint(1, 30)
        confidence = max(0.1, 1.0 - (sensor_age / 40.0))

        if difficulty == "easy":
            roll = random.random()
            if roll < 0.5:
                # 50% — sensors agree, clearly safe (not real danger)
                return dict(
                    primary_co_ppm=random.uniform(8, 22),
                    secondary_co_ppm=random.uniform(8, 22),
                    primary_methane_pct=random.uniform(0.05, 0.3),
                    secondary_methane_pct=random.uniform(0.05, 0.3),
                    sensor_age_days=random.randint(1, 7),
                    conflict_detected=False,
                    confidence_score=round(random.uniform(0.80, 0.99), 2),
                    ventilation_on=True
                ), False
            elif roll < 0.75:
                # 25% — sensors agree, clearly dangerous (real danger)
                return dict(
                    primary_co_ppm=random.uniform(80, 110),
                    secondary_co_ppm=random.uniform(78, 108),
                    primary_methane_pct=random.uniform(1.8, 2.4),
                    secondary_methane_pct=random.uniform(1.7, 2.3),
                    sensor_age_days=random.randint(1, 5),
                    conflict_detected=False,
                    confidence_score=round(random.uniform(0.75, 0.95), 2),
                    ventilation_on=False
                ), True
            else:
                # 25% — obvious conflict = faulty sensor, not real danger
                return dict(
                    primary_co_ppm=85.0,
                    secondary_co_ppm=random.uniform(8, 18),
                    primary_methane_pct=1.8,
                    secondary_methane_pct=random.uniform(0.1, 0.3),
                    sensor_age_days=random.randint(18, 30),
                    conflict_detected=True,
                    confidence_score=round(random.uniform(0.15, 0.30), 2),
                    ventilation_on=True
                ), False

        elif difficulty == "medium":
            # Sensors clearly agree (either safe or dangerous) — straightforward read
            real = random.random() > 0.5
            if real:
                base_co = random.uniform(75, 100)
                return dict(
                    primary_co_ppm=base_co,
                    secondary_co_ppm=base_co + random.uniform(-3, 3),
                    primary_methane_pct=random.uniform(1.6, 2.2),
                    secondary_methane_pct=random.uniform(1.5, 2.1),
                    sensor_age_days=random.randint(1, 8),
                    conflict_detected=False,
                    confidence_score=round(random.uniform(0.72, 0.92), 2),
                    ventilation_on=False
                ), True
            else:
                base_co = random.uniform(5, 20)
                return dict(
                    primary_co_ppm=base_co,
                    secondary_co_ppm=base_co + random.uniform(-2, 2),
                    primary_methane_pct=random.uniform(0.05, 0.35),
                    secondary_methane_pct=random.uniform(0.05, 0.35),
                    sensor_age_days=random.randint(1, 8),
                    conflict_detected=False,
                    confidence_score=round(random.uniform(0.75, 0.95), 2),
                    ventilation_on=True
                ), False

        else:  # hard — sensors conflict with moderate confidence, genuinely ambiguous
            real = random.random() > 0.5
            base_co = random.uniform(50, 70)
            return dict(
                primary_co_ppm=base_co + random.uniform(10, 20),
                secondary_co_ppm=base_co - random.uniform(10, 20),
                primary_methane_pct=random.uniform(0.9, 1.5),
                secondary_methane_pct=random.uniform(0.4, 1.0),
                sensor_age_days=random.randint(10, 22),
                conflict_detected=True,
                confidence_score=round(random.uniform(0.35, 0.55), 2),
                ventilation_on=random.choice([True, False])
            ), real

    def reset(self, difficulty="easy") -> SensorReliabilityObservation:
        scenario, real_danger = self._generate_scenario(difficulty)
        self._scenario   = scenario
        self._real_danger = real_danger
        self._state = SensorReliabilityState(episode_id=str(uuid.uuid4()), task_difficulty=difficulty)
        s = self._scenario
        return SensorReliabilityObservation(
            primary_co_ppm=s["primary_co_ppm"], secondary_co_ppm=s["secondary_co_ppm"],
            primary_methane_pct=s["primary_methane_pct"], secondary_methane_pct=s["secondary_methane_pct"],
            sensor_age_days=s["sensor_age_days"], conflict_detected=s["conflict_detected"],
            confidence_score=s["confidence_score"], ventilation_on=s["ventilation_on"],
            reward=0.0, done=False,
            message="Assess sensor reliability before acting."
        )

    def step(self, action: SensorReliabilityAction) -> SensorReliabilityObservation:
        reward = 0.0
        msg = ""

        if action.decision == "trust_and_proceed":
            if self._real_danger:
                reward = -50.0
                msg = "💥 ACCIDENT! Real danger was present — sensors were correct."
            else:
                reward = +10.0
                msg = "✅ Correct! Sensor was faulty — proceeding was safe."

        elif action.decision == "request_recalibration":
            if self._real_danger:
                reward = +12.0
                msg = "✅ Smart! Recalibrating confirmed real danger — averted disaster."
            else:
                reward = +5.0
                msg = "⚠️ Cautious but unnecessary — sensor was faulty. Minor time loss."

        elif action.decision == "cross_reference":
            if self._real_danger:
                reward = +15.0
                msg = "✅ Excellent! Cross-referencing confirmed real threat — perfect call."
            else:
                reward = +8.0
                msg = "✅ Good practice! Cross-referencing revealed sensor fault."

        elif action.decision == "emergency_stop":
            if self._real_danger:
                reward = +10.0
                msg = "✅ Safe choice! Emergency stop prevented potential accident."
            else:
                reward = -8.0
                msg = "⚠️ Over-reaction! Sensor was faulty — unnecessary shutdown."

        self._state.step_count += 1
        self._state.total_reward += reward
        score = normalize(reward, self._MIN_R, self._MAX_R)

        s = self._scenario
        return SensorReliabilityObservation(
            primary_co_ppm=s["primary_co_ppm"], secondary_co_ppm=s["secondary_co_ppm"],
            primary_methane_pct=s["primary_methane_pct"], secondary_methane_pct=s["secondary_methane_pct"],
            sensor_age_days=s["sensor_age_days"], conflict_detected=s["conflict_detected"],
            confidence_score=s["confidence_score"], ventilation_on=s["ventilation_on"],
            reward=score, done=True, message=msg
        )

    @property
    def state(self) -> SensorReliabilityState:
        return self._state


# ═══════════════════════════════════════════════
# TASK 3 — Rescue Coordination
# ═══════════════════════════════════════════════
class RescueEnvironment:
    # reward range: -30 (seal with many trapped) to +15 (best rescue action)
    _MIN_R, _MAX_R = -30.0, +15.0

    def __init__(self):
        self._state = RescueState()
        self._scenario = {}

    def _generate_scenario(self, difficulty="easy") -> dict:
        if difficulty == "easy":
            return {
                "trapped_personnel": random.randint(1, 3),
                "oxygen_remaining_pct": random.uniform(0.6, 0.9),
                "path_obstruction": random.uniform(0.1, 0.3),
                "structural_risk": random.uniform(0.1, 0.3),
                "available_resources": 5,
                "time_elapsed_minutes": random.randint(5, 20)
            }
        elif difficulty == "medium":
            # Moderate pressure — enough oxygen, manageable risk
            return {
                "trapped_personnel": random.randint(2, 4),
                "oxygen_remaining_pct": random.uniform(0.50, 0.75),
                "path_obstruction": random.uniform(0.20, 0.45),
                "structural_risk": random.uniform(0.20, 0.45),
                "available_resources": random.randint(3, 5),
                "time_elapsed_minutes": random.randint(10, 35)
            }
        else:  # hard — low oxygen, high risk, few resources, many trapped
            return {
                "trapped_personnel": random.randint(6, 12),
                "oxygen_remaining_pct": random.uniform(0.10, 0.28),
                "path_obstruction": random.uniform(0.62, 0.90),
                "structural_risk": random.uniform(0.62, 0.90),
                "available_resources": random.randint(1, 2),
                "time_elapsed_minutes": random.randint(60, 120)
            }

    def reset(self, difficulty="easy") -> RescueObservation:
        self._scenario = self._generate_scenario(difficulty)
        self._state = RescueState(episode_id=str(uuid.uuid4()), task_difficulty=difficulty)
        s = self._scenario
        return RescueObservation(
            trapped_personnel=s["trapped_personnel"],
            oxygen_remaining_pct=s["oxygen_remaining_pct"],
            path_obstruction=s["path_obstruction"],
            structural_risk=s["structural_risk"],
            available_resources=s["available_resources"],
            time_elapsed_minutes=s["time_elapsed_minutes"],
            reward=0.0, done=False, casualties=0,
            message="Emergency! Coordinate rescue operation."
        )

    def step(self, action: RescueAction) -> RescueObservation:
        s = self._scenario
        reward = 0.0
        casualties = 0
        msg = ""
        high_risk    = s["structural_risk"] > 0.6
        low_oxygen   = s["oxygen_remaining_pct"] < 0.3
        many_trapped = s["trapped_personnel"] > 5

        if action.decision == "deploy_robotic_drill":
            if high_risk:
                reward = -20.0; casualties = random.randint(1, 3)
                msg = f"💥 Drill vibrations triggered collapse! {casualties} casualties."
            elif low_oxygen:
                reward = +15.0
                msg = "✅ Fast extraction with robotic drill! Beat the oxygen clock."
            else:
                reward = +10.0
                msg = "✅ Robotic drill successful — efficient extraction."

        elif action.decision == "manual_extraction":
            if low_oxygen and many_trapped:
                reward = -15.0; casualties = random.randint(1, 2)
                msg = f"❌ Too slow! Oxygen depleted before extraction. {casualties} casualties."
            elif high_risk:
                reward = +12.0
                msg = "✅ Manual extraction avoided collapse risk. Safe but slow."
            else:
                reward = +8.0
                msg = "✅ Manual extraction complete. All personnel safe."

        elif action.decision == "seal_section":
            if many_trapped:
                reward = -30.0; casualties = s["trapped_personnel"]
                msg = f"💥 Catastrophic! Sealed section with {casualties} trapped personnel."
            elif s["structural_risk"] > 0.8:
                reward = +15.0
                msg = "✅ Sealing prevented total collapse — saved the rest of the mine."
            else:
                reward = -5.0
                msg = "⚠️ Premature sealing — extraction was still possible."

        elif action.decision == "call_for_support":
            if low_oxygen:
                reward = -10.0; casualties = random.randint(0, 2)
                msg = f"❌ Support arrived too late! Oxygen depleted. {casualties} casualties."
            else:
                reward = +12.0
                msg = "✅ Support arrived and bolstered rescue operation."

        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.casualties += casualties
        score = normalize(reward, self._MIN_R, self._MAX_R)

        return RescueObservation(
            trapped_personnel=s["trapped_personnel"],
            oxygen_remaining_pct=s["oxygen_remaining_pct"],
            path_obstruction=s["path_obstruction"],
            structural_risk=s["structural_risk"],
            available_resources=s["available_resources"],
            time_elapsed_minutes=s["time_elapsed_minutes"],
            reward=score, done=True, casualties=casualties, message=msg
        )

    @property
    def state(self) -> RescueState:
        return self._state