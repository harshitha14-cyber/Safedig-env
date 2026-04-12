import uuid
import random
from server.models import SafeDigAction, SafeDigObservation, SafeDigState

THRESHOLDS = {
    "gas_co_ppm":       {"safe": 35,  "danger": 70},
    "gas_h2s_ppm":      {"safe": 5,   "danger": 20},
    "methane_pct":      {"safe": 0.5, "danger": 1.5},
    "roof_stability":   {"safe": 0.7, "danger": 0.4},
    "earthquake_risk":  {"safe": 0.2, "danger": 0.6},
}

class SafeDigEnvironment:
    def __init__(self):
        self._state = SafeDigState()
        self._obs = None
        self._current_scenario = {}

    def _generate_scenario(self, difficulty="easy") -> dict:
        """Generate a sensor snapshot. Hard scenarios have borderline values."""
        if difficulty == "easy":
            # Clearly safe OR clearly dangerous
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
        else:  # hard — borderline everything
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
        return SafeDigObservation(
            **self._current_scenario,
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

        # Score out of 1.0 for OpenEnv grader (normalize from [-50, +15] range)
        score = (reward + 50) / 65.0
        score = max(0.0, min(1.0, score))

        return SafeDigObservation(
            **s,
            reward=score,
            done=done,
            accident_occurred=accident,
            message=msg
        )

    @property
    def state(self) -> SafeDigState:
        """Get current environment state."""
        return self._state