import uuid, random
from models import SafeDigAction, SafeDigObservation, SafeDigState

THRESHOLDS = {
    "gas_co_ppm":       {"safe": 35,  "danger": 70},
    "gas_h2s_ppm":      {"safe": 5,   "danger": 20},
    "methane_pct":      {"safe": 0.5, "danger": 1.5},
    "roof_stability":   {"safe": 0.7, "danger": 0.4},  # lower = worse
    "earthquake_risk":  {"safe": 0.2, "danger": 0.6},  # higher = worse
}

class SafeDigEnvironment:
    def __init__(self):
        self._state = SafeDigState()
        self._obs = None

    def _generate_scenario(self, difficulty="easy"):
        """Generate a sensor snapshot. Hard scenarios have borderline values."""
        if difficulty == "easy":
            # Clearly safe OR clearly dangerous
            safe = random.random() > 0.5
            if safe:
                return dict(gas_co_ppm=10, gas_h2s_ppm=2, methane_pct=0.1,
                            roof_stability=0.9, earthquake_risk=0.05,
                            ventilation_on=True, support_beams_ok=True,
                            last_near_miss_days=30)
            else:
                return dict(gas_co_ppm=90, gas_h2s_ppm=25, methane_pct=2.0,
                            roof_stability=0.2, earthquake_risk=0.8,
                            ventilation_on=False, support_beams_ok=False,
                            last_near_miss_days=1)

        elif difficulty == "medium":
            return dict(
                gas_co_ppm=random.uniform(20, 80),
                gas_h2s_ppm=random.uniform(3, 22),
                methane_pct=random.uniform(0.3, 1.8),
                roof_stability=random.uniform(0.35, 0.85),
                earthquake_risk=random.uniform(0.1, 0.7),
                ventilation_on=random.choice([True, False]),
                support_beams_ok=random.choice([True, True, False]),
                last_near_miss_days=random.randint(1, 15)
            )
        else:  # hard — borderline everything
            return dict(
                gas_co_ppm=random.uniform(30, 45),
                gas_h2s_ppm=random.uniform(4, 7),
                methane_pct=random.uniform(0.4, 0.7),
                roof_stability=random.uniform(0.55, 0.75),
                earthquake_risk=random.uniform(0.18, 0.32),
                ventilation_on=True,
                support_beams_ok=random.choice([True, False]),
                last_near_miss_days=random.randint(3, 10)
            )

    def _is_actually_dangerous(self, s) -> bool:
        return (
            s["gas_co_ppm"] > THRESHOLDS["gas_co_ppm"]["danger"] or
            s["gas_h2s_ppm"] > THRESHOLDS["gas_h2s_ppm"]["danger"] or
            s["methane_pct"] > THRESHOLDS["methane_pct"]["danger"] or
            s["roof_stability"] < THRESHOLDS["roof_stability"]["danger"] or
            s["earthquake_risk"] > THRESHOLDS["earthquake_risk"]["danger"] or
            not s["support_beams_ok"]
        )

    def reset(self, difficulty="easy") -> SafeDigObservation:
        self._current_scenario = self._generate_scenario(difficulty)
        self._state = SafeDigState(
            episode_id=str(uuid.uuid4()),
            task_difficulty=difficulty
        )
        return SafeDigObservation(
            **self._current_scenario,
            reward=0.0, done=False,
            accident_occurred=False,
            message="New shift starting. Assess the site."
        )

    def step(self, action: SafeDigAction) -> SafeDigObservation:
        s = self._current_scenario
        dangerous = self._is_actually_dangerous(s)
        reward = 0.0
        accident = False
        done = True

        if action.decision == "approve":
            if dangerous:
                reward = -50.0   # missed real danger
                accident = True
                self._state.accidents += 1
                msg = "💥 ACCIDENT! Conditions were dangerous but task was approved."
            else:
                reward = +10.0
                msg = "✅ Correct! Conditions were safe. Task completed."

        elif action.decision == "postpone":
            if dangerous:
                reward = +15.0
                msg = "✅ Good call! Conditions were dangerous. Postponing was right."
            else:
                reward = -5.0    # over-conservative
                msg = "⚠️ Over-cautious. Conditions were actually fine."

        elif action.decision == "scale_down":
            if dangerous:
                reward = +8.0    # partial credit — safer than approve
                msg = "⚠️ Partial: Conditions dangerous but scaled-down helps reduce risk."
            else:
                reward = +5.0
                msg = "✅ Acceptable. Minor precaution taken on safe conditions."

        elif action.decision == "mandate_safety":
            if dangerous:
                reward = +12.0   # good — added protection in danger
                msg = "✅ Smart. Mandating extra safety in risky conditions."
            else:
                reward = +3.0    # slight waste but not harmful
                msg = "⚠️ Conditions were safe, extra measures unnecessary but harmless."

        self._state.step_count += 1
        self._state.total_reward += reward

        # Score out of 1.0 for OpenEnv grader (normalize from [-50, +15] range)
        score = (reward + 50) / 65.0

        return SafeDigObservation(
            **s,
            reward=score,
            done=done,
            accident_occurred=accident,
            message=msg
        )

    @property
    def state(self) -> SafeDigState:
        return self._state