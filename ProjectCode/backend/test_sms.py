import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
from_phone = os.getenv('TWILIO_PHONE')
to_phones = os.getenv('RESCUE_TEAM_PHONES').split(',')

print(f"DEBUG: SID prefix: {account_sid[:4] if account_sid else 'None'}")
print(f"DEBUG: Auth prefix: {auth_token[:4] if auth_token else 'None'}")
print(f"DEBUG: From: {from_phone}")
print(f"DEBUG: To: {to_phones}")

client = Client(account_sid, auth_token)

for phone in to_phones:
    try:
        print(f"Attempting to send SMS to {phone}...")
        message = client.messages.create(
            body="Test alert from QuantumSwarmVLA debugger.",
            from_=from_phone,
            to=phone
        )
        print(f"SUCCESS: Message SID: {message.sid}")
    except Exception as e:
        print(f"ERROR Sending to {phone}: {str(e)}")
