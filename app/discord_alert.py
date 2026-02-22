import os
import requests
import logging

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK", "https://discord.com/api/webhooks/1474862103138402306/UQo03RbfP4LXkxVNUVn_p-ZbLZXnmkBbrbDVJqOynKDbt32pBvi-TuCwrIRZl1_FGFvB")

def send_alert(message):
    if not WEBHOOK_URL:
        logging.warning("DISCORD WEBHOOD not set, skipping alert.")
        return False
    try:
        data = {"content": message}
        response = requests.post(WEBHOOK_URL, json=data)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to send Discord alert: {e}")
        

