import requests
import json
import sys

def test_chat():
    url = "http://127.0.0.1:8000/api/chat"
    payload = {
        "message": "What is the price of INFOSYS?",
        "thread_id": "debug_test_1"
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_chat()
