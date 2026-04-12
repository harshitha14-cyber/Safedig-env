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

# ══════════════════════════════════════���════════
# TASK 1 — Static Safety Decision
# ═══════════════════════════════════════════════
class SafeDigEnvironment:
    def __init__(self):
        self._state = SafeDigState()
        self._obs = None
        self._current_scenario = {}

    def _generate_scenario(self, difficulty="easy") -> dict:
        """Generate a sensor snapshot. Hard scenarios have borderline values."""
        if difficulty == "easy":
            safe = random.random() > 0.5
            if safe:
                return {
                    "gas_co_ppm": 10.0,
                    "gas_h2s_ppm": 2.0,
                    "methane_pct": 0.1,
                    "roof_stability": 0.9,
                    "earthquake_risk": 0.05,
                    "ventilation_on": True,
                    "support_beams_ok": True,
                    "last_near_miss_days": 30
                }
            else:
                return {
                    "gas_co_ppm": 90.0,
                    "gas_h2s_ppm": 25.0,
                    "methane_pct": 2.0,
                    "roof_stability": 0.2,
                    "earthquake_risk": 0.8,
                    "ventilation_on": False,
                    "support_beams_ok": False,
                    "last_near_miss_days": 1
                }

        elif difficulty == "medium":
            return {
                "gas_co_ppm": random.uniform(20, 80),
                "gas_h2s_ppm": random.uniform(3, 22),
                "methane_pct": random.uniform(0.3, 1.8),
                "roof_stability": random.uniform(0.35, 0.85),
                "earthquake_risk": random.uniform(0.1, 0.7),
                "ventilation_on": random.choice([True, False]),
                "support_beams_ok": random.choice([True, True, False]),
                "last_near_miss_days": random.randint(1, 15)
            }
        else:  # hard
            return {
                "gas_co_ppm": random.uniform(30, 45),
                "gas_h2s_ppm": random.uniform(4, 7),
                "methane_pct": random.uniform(0.4, 0.7),
                "roof_stability": random.uniform(0.55, 0.75),
                "earthquake_risk": random.uniform(0.18, 0.32),
                "ventilation_on": True,
                "support_beams_ok": random.choice([True, False]),
                "last_near_miss_days": random.randint(3, 10)
            }

    def _is_actually_dangerous(self, s: dict) -> bool:
        """Check if conditions are actually dangerous based on thresholds."""
        return (
            s["gas_co_ppm"] > THRESHOLDS["gas_co_ppm"]["danger"] or
            s["gas_h2s_ppm"] > THRESHOLDS["gas_h2s_ppm"]["danger"] or
            s["methane_pct"] > THRESHOLDS["methane_pct"]["danger"] or
            s["roof_stability"] < THRESHOLDS["roof_stability"]["danger"] or
            s["earthquake_risk"] > THRESHOLDS["earthquake_risk"]["danger"] or
            not s["support_beams_ok"]
        )

    def _is_caution_zone(self, s: dict) -> bool:
        """Check if conditions are in caution zone (between safe and danger)."""
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
        """Reset environment and return initial observation."""
        self._current_scenario = self._generate_scenario(difficulty)
        self._state = SafeDigState(
            episode_id=str(uuid.uuid4()),
            task_difficulty=difficulty
        )
        s = self._current_scenario
        return SafeDigObservation(
            gas_co_ppm=s["gas_co_ppm"],
            gas_h2s_ppm=s["gas_h2s_ppm"],
            methane_pct=s["methane_pct"],
            roof_stability=s["roof_stability"],
            earthquake_risk=s["earthquake_risk"],
            ventilation_on=s["ventilation_on"],
            support_beams_ok=s["support_beams_ok"],
            last_near_miss_days=s["last_near_miss_days"],
            reward=0.0,
            done=False,
            accident_occurred=False,
            message="New shift starting. Assess the site."
        )

    def step(self, action: SafeDigAction) -> SafeDigObservation:
        """Take a step in the environment."""
        s = self._current_scenario
        dangerous = self._is_actually_dangerous(s)
        caution = self._is_caution_zone(s)
        reward = 0.0
        accident = False
        done = True
        msg = ""

        if action.decision == "approve":
            if dangerous:
                reward = -50.0
                accident = True
                self._state.accidents += 1
                msg = "💥 ACCIDENT! Conditions were dangerous but task was approved."
            elif caution:
                reward = -10.0
                msg = "⚠️ WARNING! Conditions are borderline (caution zone). Should have postponed or added safety measures."
            else:
                reward = +10.0
                msg = "✅ Perfect! Conditions were safe. Task approved and completed successfully."

        elif action.decision == "postpone":
            if dangerous:
                reward = +15.0
                msg = "✅ Excellent! Conditions were dangerous. Postponing was the right decision."
            elif caution:
                reward = +8.0
                msg = "✅ Good call! Conditions are borderline - waiting is prudent."
            else:
                reward = -5.0
                msg = "⚠️ Over-cautious. Conditions were actually safe. Consider approving next time."

        elif action.decision == "scale_down":
            if dangerous:
                reward = +8.0
                msg = "⚠️ Acceptable: Conditions dangerous but scaling down reduces risk. Postponing would be better."
            elif caution:
                reward = +5.0
                msg = "✅ Reasonable compromise. Conditions are borderline, scaling down is acceptable."
            else:
                reward = -2.0
                msg = "⚠️ Unnecessary reduction. Conditions were safe, no need to scale down operations."

        elif action.decision == "mandate_safety":
            if dangerous:
                reward = +12.0
                msg = "✅ Excellent! Mandating extra safety measures in dangerous conditions is very responsible."
            elif caution:
                reward = +6.0
                msg = "✅ Good precaution. Extra safety measures in caution zone is wise but slightly overkill."
            else:
                reward = -3.0
                msg = "⚠️ Unnecessary expense. Conditions were completely safe, extra safety measures weren't needed."

        self._state.step_count += 1
        self._state.total_reward += reward
        score = max(0.0, min(1.0, (reward + 50) / 65.0))

        return SafeDigObservation(
            gas_co_ppm=s["gas_co_ppm"],
            gas_h2s_ppm=s["gas_h2s_ppm"],
            methane_pct=s["methane_pct"],
            roof_stability=s["roof_stability"],
            earthquake_risk=s["earthquake_risk"],
            ventilation_on=s["ventilation_on"],
            support_beams_ok=s["support_beams_ok"],
            last_near_miss_days=s["last_near_miss_days"],
            reward=score,
            done=done,
            accident_occurred=accident,
            message=msg
        )

    @property
    def state(self) -> SafeDigState:
        """Get current environment state."""
        return self._state


# ═══════════════════════════════════════════════
# TASK 2 — Sensor Reliability & Ghost Hazards
# ═══════════════════════════════════════════════
class SensorReliabilityEnvironment:
    def __init__(self):
        self._state = SensorReliabilityState()
        self._scenario = {}
        self._real_danger = False

    def _generate_scenario(self, difficulty="easy"):
        """Generate sensor readings with possible conflicts."""
        sensor_age = random.randint(1, 30)
        confidence = max(0.1, 1.0 - (sensor_age / 40.0))

        if difficulty == "easy":
            if random.random() > 0.5:
                return dict(
                    primary_co_ppm=85.0,
                    secondary_co_ppm=82.0,
                    primary_methane_pct=1.8,
                    secondary_methane_pct=1.7,
                    sensor_age_days=int(sensor_age),
                    conflict_detected=False,
                    confidence_score=float(confidence),
                    ventilation_on=False
                ), True
            else:
                return dict(
                    primary_co_ppm=85.0, secondary_co_ppm=12.0,
                    primary_methane_pct=1.8, secondary_methane_pct=0.2,
                    sensor_age_days=int(sensor_age + 15),
                    conflict_detected=True,
                    confidence_score=0.2, ventilation_on=True
                ), False

        elif difficulty == "medium":
            real = random.random() > 0.5
            noise = random.uniform(5, 20)
            base_co = random.uniform(40, 75) if real else random.uniform(15, 35)
            return dict(
                primary_co_ppm=base_co + (noise if real else -noise * 0.5),
                secondary_co_ppm=base_co + random.uniform(-noise, noise),
                primary_methane_pct=random.uniform(0.8, 1.6) if real else random.uniform(0.1, 0.5),
                secondary_methane_pct=random.uniform(0.5, 1.4) if real else random.uniform(0.1, 0.7),
                sensor_age_days=int(sensor_age),
                conflict_detected=random.random() > 0.5,
                confidence_score=float(confidence),
                ventilation_on=random.choice([True, False])
            ), real

        else:  # hard
            real = random.random() > 0.5
            return dict(
                primary_co_ppm=random.uniform(55, 72),
                secondary_co_ppm=random.uniform(48, 75),
                primary_methane_pct=random.uniform(0.9, 1.6),
                secondary_methane_pct=random.uniform(0.7, 1.5),
                sensor_age_days=int(random.randint(10, 25)),
                conflict_detected=True,
                confidence_score=float(random.uniform(0.3, 0.6)),
                ventilation_on=random.choice([True, False])
            ), real

    def reset(self, difficulty="easy") -> SensorReliabilityObservation:
        """Reset environment and return initial observation."""
        scenario, real_danger = self._generate_scenario(difficulty)
        self._scenario = scenario
        self._real_danger = real_danger
        self._state = SensorReliabilityState(
            episode_id=str(uuid.uuid4()),
            task_difficulty=difficulty
        )
        s = self._scenario
        return SensorReliabilityObservation(
            primary_co_ppm=s["primary_co_ppm"],
            secondary_co_ppm=s["secondary_co_ppm"],
            primary_methane_pct=s["primary_methane_pct"],
            secondary_methane_pct=s["secondary_methane_pct"],
            sensor_age_days=s["sensor_age_days"],
            conflict_detected=s["conflict_detected"],
            confidence_score=s["confidence_score"],
            ventilation_on=s["ventilation_on"],
            reward=0.0,
            done=False,
            message="Conflicting sensor readings detected. Assess reliability."
        )

    def step(self, action: SensorReliabilityAction) -> SensorReliabilityObservation:
        """Take a step in the environment."""
        reward = 0.0
        msg = ""
        done = True

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
        score = max(0.0, min(1.0, (reward + 50) / 65.0))

        s = self._scenario
        return SensorReliabilityObservation(
            primary_co_ppm=s["primary_co_ppm"],
            secondary_co_ppm=s["secondary_co_ppm"],
            primary_methane_pct=s["primary_methane_pct"],
            secondary_methane_pct=s["secondary_methane_pct"],
            sensor_age_days=s["sensor_age_days"],
            conflict_detected=s["conflict_detected"],
            confidence_score=s["confidence_score"],
            ventilation_on=s["ventilation_on"],
            reward=score,
            done=done,
            message=msg
        )

    @property
    def state(self) -> SensorReliabilityState:
        """Get current environment state."""
        return self._state


# ═══════════════════════════════════════════════
# TASK 3 — Rescue Coordination
# ═══════════════════════════════════════════════
class RescueEnvironment:
    def __init__(self):
        self._state = RescueState()
        self._scenario = {}

    def _generate_scenario(self, difficulty="easy") -> dict:
        """Generate a rescue emergency scenario."""
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
            return {
                "trapped_personnel": random.randint(3, 7),
                "oxygen_remaining_pct": random.uniform(0.3, 0.6),
                "path_obstruction": random.uniform(0.3, 0.6),
                "structural_risk": random.uniform(0.3, 0.6),
                "available_resources": random.randint(2, 4),
                "time_elapsed_minutes": random.randint(20, 60)
            }
        else:  # hard
            return {
                "trapped_personnel": random.randint(6, 12),
                "oxygen_remaining_pct": random.uniform(0.1, 0.3),
                "path_obstruction": random.uniform(0.6, 0.9),
                "structural_risk": random.uniform(0.6, 0.9),
                "available_resources": random.randint(1, 2),
                "time_elapsed_minutes": random.randint(60, 120)
            }

    def reset(self, difficulty="easy") -> RescueObservation:
        """Reset environment and return initial observation."""
        self._scenario = self._generate_scenario(difficulty)
        self._state = RescueState(
            episode_id=str(uuid.uuid4()),
            task_difficulty=difficulty
        )
        s = self._scenario
        return RescueObservation(
            trapped_personnel=s["trapped_personnel"],
            oxygen_remaining_pct=s["oxygen_remaining_pct"],
            path_obstruction=s["path_obstruction"],
            structural_risk=s["structural_risk"],
            available_resources=s["available_resources"],
            time_elapsed_minutes=s["time_elapsed_minutes"],
            reward=0.0,
            done=False,
            casualties=0,
            message="Emergency! Coordinate rescue operation."
        )

    def step(self, action: RescueAction) -> RescueObservation:
        """Take a step in the environment."""
        s = self._scenario
        reward = 0.0
        casualties = 0
        msg = ""
        done = True
        high_risk = s["structural_risk"] > 0.6
        low_oxygen = s["oxygen_remaining_pct"] < 0.3
        many_trapped = s["trapped_personnel"] > 5

        if action.decision == "deploy_robotic_drill":
            if high_risk:
                reward = -20.0
                casualties = random.randint(1, 3)
                msg = f"💥 Drill vibrations triggered collapse! {casualties} casualties."
            elif low_oxygen:
                reward = +15.0
                msg = "✅ Fast extraction with robotic drill! Beat the oxygen clock."
            else:
                reward = +10.0
                msg = "✅ Robotic drill successful — efficient extraction."

        elif action.decision == "manual_extraction":
            if low_oxygen and many_trapped:
                reward = -15.0
                casualties = random.randint(1, 2)
                msg = f"❌ Too slow! Oxygen depleted before extraction. {casualties} casualties."
            elif high_risk:
                reward = +12.0
                msg = "✅ Manual extraction avoided collapse risk. Safe but slow."
            else:
                reward = +8.0
                msg = "✅ Manual extraction complete. All personnel safe."

        elif action.decision == "seal_section":
            if many_trapped:
                reward = -30.0
                casualties = s["trapped_personnel"]
                msg = f"💥 Catastrophic! Sealed section with {casualties} trapped personnel."
            elif s["structural_risk"] > 0.8:
                reward = +15.0
                msg = "✅ Sealing prevented total collapse — saved the rest of the mine."
            else:
                reward = -5.0
                msg = "⚠️ Premature sealing — extraction was still possible."

        elif action.decision == "call_for_support":
            if low_oxygen:
                reward = -10.0
                casualties = random.randint(0, 2)
                msg = f"❌ Support arrived too late! Oxygen depleted. {casualties} casualties."
            else:
                reward = +12.0
                msg = "✅ Support arrived and bolstered rescue operation."

        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.casualties += casualties
        score = max(0.0, min(1.0, (reward + 50) / 80.0))

        return RescueObservation(
            trapped_personnel=s["trapped_personnel"],
            oxygen_remaining_pct=s["oxygen_remaining_pct"],
            path_obstruction=s["path_obstruction"],
            structural_risk=s["structural_risk"],
            available_resources=s["available_resources"],
            time_elapsed_minutes=s["time_elapsed_minutes"],
            reward=score,
            done=done,
            casualties=casualties,
            message=msg
        )

    @property
    def state(self) -> RescueState:
        """Get current environment state."""
        return self._state