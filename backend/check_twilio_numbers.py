import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')

try:
    print("Authenticating with Twilio...")
    client = Client(account_sid, auth_token)
    
    print("Fetching available sender numbers on this account...")
    incoming_numbers = client.incoming_phone_numbers.list(limit=5)
    
    if incoming_numbers:
        print("\n✅ Found valid Twilio numbers for this account:")
        for record in incoming_numbers:
            print(f" - {record.phone_number} ({record.friendly_name})")
    else:
        print("\n❌ No phone numbers found on this account. You must purchase/provision one to send SMS.")

except Exception as e:
    print(f"\n❌ Error fetching numbers: {e}")
