import os
import requests

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK", "https://discord.com/api/webhooks/1474862103138402306/UQo03RbfP4LXkxVNUVn_p-ZbLZXnmkBbrbDVJqOynKDbt32pBvi-TuCwrIRZl1_FGFvB")

def send_alert(message):
    data={"content": message}
    requests.post(WEBHOOK_URL,json=data)

