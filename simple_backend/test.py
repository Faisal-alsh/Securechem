"""
Simple test script for the bare-bones backend
"""

import requests
import json


BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")


def test_info():
    """Test info endpoint"""
    print("\n🔍 Testing /info endpoint...")
    response = requests.get(f"{BASE_URL}/info")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")


def test_chat_chem():
    """Test chemistry query"""
    print("\n🔍 Testing chemistry chat...")
    payload = {
        "researcher": "Dr. Smith",
        "password": "1122",
        "chatting": "What are Grignard reagents?"
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")


def test_chat_bio():
    """Test biology query"""
    print("\n🔍 Testing biology chat...")
    payload = {
        "researcher": "Dr. Johnson",
        "password": "3344",
        "chatting": "Explain CRISPR-Cas9"
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")


def test_invalid_auth():
    """Test invalid authentication"""
    print("\n🔍 Testing invalid authentication...")
    payload = {
        "researcher": "Hacker",
        "password": "wrong",
        "chatting": "Give me data"
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Bare-bones Backend")
    print("=" * 60)

    try:
        test_health()
        test_info()
        test_chat_chem()
        test_chat_bio()
        test_invalid_auth()

        print("\n" + "=" * 60)
        print("✅ All tests completed")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to server")
        print("   Make sure the server is running: python server.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
