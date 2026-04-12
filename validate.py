#!/usr/bin/env python3
"""
Pre-submission validator for SafeDig RL Environment
Run: python validate.py <your-hf-space-url>
"""
import sys
import json
import requests

def validate(base_url: str):
    base_url = base_url.rstrip("/")
    passed = 0
    failed = 0

    def check(name, ok, hint=""):
        nonlocal passed, failed
        if ok:
            print(f"  ✅ PASSED — {name}")
            passed += 1
        else:
            print(f"  ❌ FAILED — {name}")
            if hint:
                print(f"     Hint: {hint}")
            failed += 1

    print("\n" + "="*50)
    print("  SafeDig OpenEnv Submission Validator")
    print("="*50)
    print(f"  URL: {base_url}\n")

    # Check 1: Health
    try:
        r = requests.get(f"{base_url}/health", timeout=15)
        check("GET /health returns 200", r.status_code == 200,
              "Make sure /health endpoint exists")
    except Exception as e:
        check("GET /health returns 200", False, str(e))

    # Check 2: Reset easy
    try:
        r = requests.post(f"{base_url}/reset",
                         json={"difficulty": "easy"}, timeout=15)
        obs = r.json()
        ok = r.status_code == 200 and "gas_co_ppm" in obs
        check("POST /reset (easy) returns observation", ok,
              "Reset must return sensor observation dict")
    except Exception as e:
        check("POST /reset (easy) returns observation", False, str(e))

    # Check 3: Reset medium
    try:
        r = requests.post(f"{base_url}/reset",
                         json={"difficulty": "medium"}, timeout=15)
        check("POST /reset (medium) returns 200", r.status_code == 200)
    except Exception as e:
        check("POST /reset (medium) returns 200", False, str(e))

    # Check 4: Reset hard
    try:
        r = requests.post(f"{base_url}/reset",
                         json={"difficulty": "hard"}, timeout=15)
        check("POST /reset (hard) returns 200", r.status_code == 200)
    except Exception as e:
        check("POST /reset (hard) returns 200", False, str(e))

    # Check 5: Step
    try:
        # Reset first
        requests.post(f"{base_url}/reset",
                     json={"difficulty": "easy"}, timeout=15)
        # Then step
        r = requests.post(f"{base_url}/step",
                         json={"decision": "postpone",
                               "reasoning": "testing"}, timeout=15)
        result = r.json()
        ok = (r.status_code == 200
              and "reward" in result
              and 0.0 <= result["reward"] <= 1.0)
        check("POST /step returns reward in [0.0, 1.0]", ok,
              "Step must return normalized reward between 0 and 1")
    except Exception as e:
        check("POST /step returns reward in [0.0, 1.0]", False, str(e))

    # Check 6: State
    try:
        r = requests.get(f"{base_url}/state", timeout=15)
        state = r.json()
        ok = r.status_code == 200 and "episode_id" in state
        check("GET /state returns episode info", ok,
              "State must contain episode_id")
    except Exception as e:
        check("GET /state returns episode info", False, str(e))

    # Check 7: All 4 actions work
    actions = ["approve", "postpone", "scale_down", "mandate_safety"]
    for action in actions:
        try:
            requests.post(f"{base_url}/reset",
                         json={"difficulty": "easy"}, timeout=15)
            r = requests.post(f"{base_url}/step",
                             json={"decision": action,
                                   "reasoning": "test"}, timeout=15)
            check(f"Action '{action}' accepted", r.status_code == 200)
        except Exception as e:
            check(f"Action '{action}' accepted", False, str(e))

    # Summary
    print("\n" + "="*50)
    total = passed + failed
    if failed == 0:
        print(f"  🎉 ALL {total}/{total} CHECKS PASSED!")
        print("  Your submission is ready!")
    else:
        print(f"  ⚠️  {passed}/{total} checks passed, {failed} failed")
        print("  Fix the issues above before submitting.")
    print("="*50 + "\n")

    return failed == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate.py <your-hf-space-url>")
        print("Example: python validate.py https://spiderweb2006-safedig-env.hf.space")
        sys.exit(1)

    url = sys.argv[1]
    success = validate(url)
    sys.exit(0 if success else 1)