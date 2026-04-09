from fastapi import FastAPI
from pydantic import BaseModel
import random
from typing import List, Dict, Any

app = FastAPI(title="SafeDig Environment API")

class Action(BaseModel):
    action: int  # 0=approve, 1=postpone, 2=scale_down, 3=mandate_safety

class ResetRequest(BaseModel):
    difficulty: str = "easy"

class EnvironmentState:
    def __init__(self):
        self.reset("easy")
    
    def reset(self, difficulty: str = "easy"):
        if difficulty == "easy":
            self.gas_co_ppm = random.uniform(5, 20)
            self.gas_h2s_ppm = random.uniform(2, 8)
            self.methane_pct = random.uniform(0.1, 0.5)
            self.roof_stability = random.uniform(0.7, 0.95)
            self.earthquake_risk = random.uniform(0.1, 0.3)
        elif difficulty == "medium":
            self.gas_co_ppm = random.uniform(15, 40)
            self.gas_h2s_ppm = random.uniform(8, 15)
            self.methane_pct = random.uniform(0.3, 1.0)
            self.roof_stability = random.uniform(0.5, 0.8)
            self.earthquake_risk = random.uniform(0.2, 0.5)
        else:  # hard
            self.gas_co_ppm = random.uniform(30, 70)
            self.gas_h2s_ppm = random.uniform(12, 25)
            self.methane_pct = random.uniform(0.8, 1.8)
            self.roof_stability = random.uniform(0.3, 0.6)
            self.earthquake_risk = random.uniform(0.4, 0.8)
        
        self.ventilation_on = True
        self.support_beams_ok = True
        self.last_near_miss_days = random.randint(1, 30)
        self.step_count = 0
        self.accidents = 0
        self.difficulty = difficulty
        self.total_reward = 0.0
        
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        return {
            "gas_co_ppm": round(self.gas_co_ppm, 2),
            "gas_h2s_ppm": round(self.gas_h2s_ppm, 2),
            "methane_pct": round(self.methane_pct, 2),
            "roof_stability": round(self.roof_stability, 2),
            "earthquake_risk": round(self.earthquake_risk, 2),
            "ventilation_on": self.ventilation_on,
            "support_beams_ok": self.support_beams_ok,
            "last_near_miss_days": self.last_near_miss_days,
            "step_count": self.step_count,
            "accidents": self.accidents,
            "difficulty": self.difficulty,
            "message": f"Environment ready - {self.difficulty.upper()} difficulty"
        }
    
    def calculate_risk_score(self) -> float:
        risks = []
        if self.gas_co_ppm > 70: risks.append(1.0)
        elif self.gas_co_ppm > 35: risks.append(0.7)
        else: risks.append(0.2)
        
        if self.gas_h2s_ppm > 20: risks.append(1.0)
        elif self.gas_h2s_ppm > 5: risks.append(0.7)
        else: risks.append(0.2)
        
        if self.methane_pct > 1.5: risks.append(1.0)
        elif self.methane_pct > 0.5: risks.append(0.7)
        else: risks.append(0.2)
        
        if self.roof_stability < 0.4: risks.append(1.0)
        elif self.roof_stability < 0.7: risks.append(0.7)
        else: risks.append(0.2)
        
        if self.earthquake_risk > 0.6: risks.append(1.0)
        elif self.earthquake_risk > 0.2: risks.append(0.7)
        else: risks.append(0.2)
        
        if not self.ventilation_on: risks.append(0.8)
        if not self.support_beams_ok: risks.append(0.9)
        if self.last_near_miss_days < 7: risks.append(0.6)
        
        return sum(risks) / len(risks) if risks else 0.5
    
    def step(self, action: int):
        action_names = ["approve", "postpone", "scale_down", "mandate_safety"]
        action_name = action_names[action] if 0 <= action < 4 else "approve"
        
        risk_score = self.calculate_risk_score()
        accident_occurred = False
        reward = 0
        message = ""
        
        if action_name == "approve":
            if risk_score < 0.4:
                reward = 15
                message = "✅ Safe conditions approved. Operation proceeds smoothly."
            elif risk_score < 0.7:
                reward = 5
                message = "⚠️ Moderate risk approved. Some issues arise but manageable."
            else:
                accident_chance = risk_score * (1.5 if self.difficulty == "hard" else 1.0)
                if random.random() < accident_chance:
                    reward = -50
                    accident_occurred = True
                    message = "🚨 ACCIDENT! High-risk approval led to disaster."
                    self.accidents += 1
                else:
                    reward = -10
                    message = "❌ High-risk approval barely avoided accident."
        
        elif action_name == "postpone":
            if risk_score > 0.6:
                reward = 20
                message = "✅ Excellent decision! Postponing avoided major risks."
            elif risk_score > 0.3:
                reward = 10
                message = "⚠️ Postponing was reasonable."
            else:
                reward = -5
                message = "❌ Unnecessary postponement wasted resources."
        
        elif action_name == "scale_down":
            if risk_score > 0.5:
                reward = 12
                message = "✅ Scaling down reduced risk."
            else:
                reward = 5
                message = "ℹ️ Scaled down operations. Safe but less efficient."
        
        elif action_name == "mandate_safety":
            reward = 8 if risk_score > 0.4 else 3
            message = "🛡️ Additional safety measures implemented."
            self.ventilation_on = True
            self.support_beams_ok = True
        
        self.step_count += 1
        self.total_reward += reward
        
        self.gas_co_ppm += random.uniform(-5, 8)
        self.gas_h2s_ppm += random.uniform(-2, 5)
        self.methane_pct += random.uniform(-0.2, 0.3)
        self.roof_stability -= random.uniform(0, 0.05)
        self.earthquake_risk += random.uniform(-0.1, 0.1)
        
        self.gas_co_ppm = max(0, min(100, self.gas_co_ppm))
        self.gas_h2s_ppm = max(0, min(50, self.gas_h2s_ppm))
        self.methane_pct = max(0, min(3.0, self.methane_pct))
        self.roof_stability = max(0, min(1.0, self.roof_stability))
        self.earthquake_risk = max(0, min(1.0, self.earthquake_risk))
        
        done = (self.step_count >= 20) or (accident_occurred and self.accidents >= 3)
        
        return {
            "observation": self.get_observation(),
            "reward": reward,
            "total_reward": self.total_reward,
            "message": message,
            "accident_occurred": accident_occurred,
            "done": done,
            "step_count": self.step_count,
            "accidents": self.accidents
        }

env = EnvironmentState()

@app.get("/info")
async def info():
    return {
        "name": "SafeDig Environment",
        "version": "1.0.0",
        "actions": ["approve", "postpone", "scale_down", "mandate_safety"],
        "observation_space": 8,
        "action_space": 4
    }

@app.post("/reset")
async def reset(request: ResetRequest):
    obs = env.reset(difficulty=request.difficulty)
    return obs

@app.get("/state")
async def state():
    return env.get_observation()

@app.post("/step")
async def step(action: Action):
    result = env.step(action.action)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)