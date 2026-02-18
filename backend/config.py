import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    
    NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN', 'YOUR_TOKEN')
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'YOUR_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'YOUR_TOKEN')
    TWILIO_PHONE = os.getenv('TWILIO_PHONE', '+1234567890')
    RESCUE_TEAM_PHONES = os.getenv('RESCUE_TEAM_PHONES', '+919876543210').split(',')
    
    N_QUBITS = int(os.getenv('N_QUBITS', 4))
    N_AGENTS = int(os.getenv('N_AGENTS', 50))
    
    DISASTER_THRESHOLD = float(os.getenv('DISASTER_THRESHOLD', 0.8))
    ALERT_CONFIDENCE_THRESHOLD = float(os.getenv('ALERT_CONFIDENCE_THRESHOLD', 0.75))
    
    DEVICE = os.getenv('DEVICE', 'cuda')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))
    TESTING_MODE = os.getenv('TESTING_MODE', 'True') == 'True'
