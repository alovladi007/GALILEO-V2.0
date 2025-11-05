#!/usr/bin/env python3
"""
Integration Test Suite
Tests all major platform capabilities end-to-end
"""

import requests
import json
import time

API_URL = "http://localhost:5050"

def test_simulation():
    """Test simulation endpoint"""
    print("\n1. Testing Simulation Service...")
    payload = {
        "initial_state": [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
        "duration_seconds": 600,
        "time_step": 60,
        "perturbations": ["J2"]
    }
    try:
        response = requests.post(f"{API_URL}/api/simulation/propagate-single", json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: Got {len(data.get('states', []))} trajectory points")
            return True
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_inversion():
    """Test inversion endpoint"""
    print("\n2. Testing Inversion Service...")
    payload = {
        "observations": [[0, 0, 1.0], [1, 0, 1.1], [0, 1, 0.9]],
        "baseline_vectors": [[1, 0, 0], [0, 1, 0]],
        "grid_resolution": 10,
        "method": "variational"
    }
    try:
        response = requests.post(f"{API_URL}/api/inversion/estimate-gravity", json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: Estimated gravity field")
            return True
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_control():
    """Test control endpoint"""
    print("\n3. Testing Control Service...")
    payload = {
        "current_state": [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
        "target_separation": 100.0,
        "time_horizon": 3600
    }
    try:
        response = requests.post(f"{API_URL}/api/control/station-keeping", json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: Generated {len(data.get('maneuvers', []))} maneuvers")
            return True
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_ml():
    """Test ML endpoint"""
    print("\n4. Testing ML Service...")
    payload = {
        "model_id": "test_pinn",
        "hidden_layers": [64, 128, 64],
        "activation": "tanh"
    }
    try:
        response = requests.post(f"{API_URL}/api/ml/pinn/create", json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: Created PINN model with {data.get('n_parameters', 0)} parameters")
            return True
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_workflow():
    """Test workflow endpoint"""
    print("\n5. Testing Workflow Service...")
    payload = {
        "workflow_type": "simulation_only",
        "parameters": {"n_satellites": 2, "duration_days": 1},
        "user_id": "test-user",
        "priority": 5
    }
    try:
        response = requests.post(f"{API_URL}/api/workflow/submit", json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: Submitted workflow {data.get('workflow_id', 'unknown')}")
            return True
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_task():
    """Test task endpoint"""
    print("\n6. Testing Task Service...")
    payload = {
        "task_name": "simulation.propagate_formation",
        "parameters": {"n_satellites": 2, "duration_days": 1},
        "priority": 5
    }
    try:
        response = requests.post(f"{API_URL}/api/tasks/submit", json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: Submitted task {data.get('task_id', 'unknown')}")
            return True
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_database():
    """Test database endpoint"""
    print("\n7. Testing Database Service...")
    payload = {
        "job_type": "simulation",
        "user_id": "550e8400-e29b-41d4-a716-446655440000",
        "config": {"test": "data"}
    }
    try:
        response = requests.post(f"{API_URL}/api/db/jobs", json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: Created job {data.get('job_id', 'unknown')}")
            return True
        else:
            print(f"   ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("="*70)
    print("GALILEO V2.0 - Integration Test Suite")
    print("="*70)

    results = {
        "Simulation": test_simulation(),
        "Inversion": test_inversion(),
        "Control": test_control(),
        "ML": test_ml(),
        "Workflow": test_workflow(),
        "Task": test_task(),
        "Database": test_database(),
    }

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for service, success in results.items():
        symbol = "✅" if success else "❌"
        print(f"{symbol} {service:20s}: {'PASS' if success else 'FAIL'}")

    print(f"\nTotal: {passed}/{total} services working ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 ALL SERVICES OPERATIONAL!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} services need attention")
        return 1

if __name__ == "__main__":
    exit(main())
