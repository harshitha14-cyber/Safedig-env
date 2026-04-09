"""
Test script for SafeDig environment
Run: python main.py
"""

import json
import subprocess
import time
import requests
import sys

# Start FastAPI server in background
print("🚀 Starting SafeDig server...")
server_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for server to start
time.sleep(3)

try:
    BASE_URL = "http://127.0.0.1:8000"
    
    print("\n" + "="*60)
    print("Testing SafeDig RL Environment")
    print("="*60)
    
    # Test health check
    print("\n✅ Health check...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {resp.status_code}")
    assert resp.status_code == 200, "Health check failed!"
    
    # Test each difficulty level
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n🎯 Testing {difficulty.upper()} difficulty...")
        
        # Reset
        reset_resp = requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty})
        assert reset_resp.status_code == 200, f"Reset failed for {difficulty}"
        obs = reset_resp.json()
        
        print(f"   Observation keys: {list(obs.keys())}")
        print(f"   Gas CO: {obs['gas_co_ppm']} ppm")
        print(f"   Roof stability: {obs['roof_stability']}")
        
        # Take a step
        action = {
            "decision": "postpone",
            "reasoning": "Testing environment"
        }
        step_resp = requests.post(f"{BASE_URL}/step", json=action)
        assert step_resp.status_code == 200, f"Step failed for {difficulty}"
        result = step_resp.json()
        
        print(f"   Reward: {result['reward']:.3f}")
        print(f"   Message: {result['message']}")
        print(f"   ✓ {difficulty.upper()} test passed!")
    
    # Test state endpoint
    print("\n📊 Testing state endpoint...")
    state_resp = requests.get(f"{BASE_URL}/state")
    assert state_resp.status_code == 200, "State endpoint failed"
    state = state_resp.json()
    print(f"   Episode ID: {state['episode_id']}")
    print(f"   Step count: {state['step_count']}")
    print(f"   Total reward: {state['total_reward']:.3f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    
finally:
    # Cleanup
    print("\n🛑 Stopping server...")
    server_process.terminate()
    server_process.wait()