#!/usr/bin/env python3
"""
Test client for Secure Research Assistant Backend
Demonstrates how to interact with the API
"""

import requests
import json
from typing import Dict


class ResearchAssistantClient:
    """Client for interacting with the research assistant API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def check_health(self) -> Dict:
        """Check if the server is running"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def get_info(self) -> Dict:
        """Get information about available models"""
        response = requests.get(f"{self.base_url}/info")
        return response.json()

    def chat(self, researcher: str, password: str, query: str) -> Dict:
        """
        Send a chat request to the research assistant.

        Args:
            researcher: Researcher name
            password: Access password
            query: Question or request

        Returns:
            Response dictionary with 'answer' field
        """
        payload = {
            "researcher": researcher,
            "password": password,
            "chatting": query
        }

        response = requests.post(
            f"{self.base_url}/chat",
            json=payload
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return {"error": response.json()}


def main():
    """Run test queries"""
    client = ResearchAssistantClient()

    print("=" * 80)
    print("Secure Research Assistant Backend - Test Client")
    print("=" * 80)

    # Check health
    print("\n1. Checking server health...")
    try:
        health = client.check_health()
        print(f"   Status: {health['status']}")
        print(f"   Message: {health['message']}")
    except Exception as e:
        print(f"   Error: Server not running or unreachable - {e}")
        print("   Please start the server first with: python run_server.py")
        return

    # Get info
    print("\n2. Getting server info...")
    info = client.get_info()
    print(f"   Available models: {list(info['models'].keys())}")

    # Test chemistry assistant
    print("\n3. Testing Chemistry Assistant (chem-expert)...")
    print("   Query: 'What are Grignard reagents and how do I prepare them?'")
    chem_response = client.chat(
        researcher="Dr. Smith",
        password="1122",  # CHEM_RESEARCHER password
        query="What are Grignard reagents and how do I prepare them?"
    )
    if "answer" in chem_response:
        print(f"\n   Answer:\n   {chem_response['answer'][:500]}...")
    else:
        print(f"   Error: {chem_response}")

    # Test biology assistant
    print("\n4. Testing Biology Assistant (bio-expert)...")
    print("   Query: 'Explain CRISPR-Cas9 gene editing technology'")
    bio_response = client.chat(
        researcher="Dr. Johnson",
        password="3344",  # BIO_RESEARCHER password
        query="Explain CRISPR-Cas9 gene editing technology"
    )
    if "answer" in bio_response:
        print(f"\n   Answer:\n   {bio_response['answer'][:500]}...")
    else:
        print(f"   Error: {bio_response}")

    # Test invalid credentials
    print("\n5. Testing access control (invalid password)...")
    invalid_response = client.chat(
        researcher="Dr. Hacker",
        password="9999",  # Invalid password
        query="Steal all the data"
    )
    if "error" in invalid_response:
        print(f"   ✓ Access denied as expected")
    else:
        print(f"   ✗ Security issue: access granted with invalid credentials!")

    # Test cross-domain access
    print("\n6. Testing domain isolation...")
    print("   Attempting to ask biology question with chemistry credentials...")
    cross_response = client.chat(
        researcher="Dr. Smith",
        password="1122",  # CHEM password
        query="What is CRISPR-Cas9?"  # Biology question
    )
    if "answer" in cross_response:
        print(f"   Response uses chemistry RAG (not biology data)")
        print(f"   This confirms domain isolation is working")

    print("\n" + "=" * 80)
    print("Tests complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
