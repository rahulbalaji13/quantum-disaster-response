import requests
import sys

image_path = r"C:\Users\rahul\.gemini\antigravity\brain\7a73d370-9fa5-493f-b100-441849399d6c\flood_disaster_test_1769620649515.png"
url = "http://localhost:5000/api/analyze"

try:
    with open(image_path, 'rb') as img:
        files = {'file': img}
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
