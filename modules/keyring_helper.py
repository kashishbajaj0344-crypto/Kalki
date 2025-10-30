"""
keyring_helper.py
Handles secure storage and retrieval of API keys using keyring.
"""

import keyring
from modules.config import OPENAI_API_KEY

SERVICE_NAME = "KalkiAI"

def get_api_key() -> str:
    """
    Retrieve the OpenAI API key from keyring or fallback to environment variable.
    """
    key = keyring.get_password(SERVICE_NAME, "openai")
    if key:
        return key
    if OPENAI_API_KEY:
        keyring.set_password(SERVICE_NAME, "openai", OPENAI_API_KEY)
        return OPENAI_API_KEY
    raise ValueError("OpenAI API key not found in keyring or environment.")

def set_api_key(key: str):
    """
    Set the OpenAI API key securely in keyring.
    """
    keyring.set_password(SERVICE_NAME, "openai", key)
