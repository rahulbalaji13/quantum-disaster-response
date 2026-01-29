import requests
import json

image_path = r"C:\Users\rahul\.gemini\antigravity\brain\7a73d370-9fa5-493f-b100-441849399d6c\flood_disaster_test_1769620649515.png"
url = "http://localhost:5000/api/analyze"

try:
    with open(image_path, 'rb') as img:
        files = {'file': img}
        response = requests.post(url, files=files)
        
    data = response.json()
    print("Disaster Type:", data.get('disaster_type'))
    print("Risk Level:", data.get('risk_level'))
    print("Alert Triggered:", data.get('alert_triggered'))
    print("Alert Status:", data.get('alert_status'))
except Exception as e:
    print(f"Error: {e}")
