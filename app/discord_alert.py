import os
import logging

import requests

from app import config


def send_alert(message):
    """Post a drift alert to Discord.

    The webhook comes from the DISCORD_WEBHOOK environment variable. There is
    deliberately no default: a URL committed to the repository is a leaked
    secret that anyone can post to.
    """
    webhook = config.DISCORD_WEBHOOK
    if not webhook:
        logging.warning("DISCORD_WEBHOOK not set, skipping alert.")
        return False
    try:
        response = requests.post(
            webhook, json={"content": message}, timeout=(3, 5)
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to send Discord alert: {e}")
        return False
