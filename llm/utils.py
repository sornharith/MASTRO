"""
Utilities for LLM integration in the multi-agent dropout prediction system
"""
import re
import json


def parse_json_from_response(text: str) -> dict:
    """Finds and parses the first valid JSON object from an LLM response string."""
    # Look for the first curly brace to the last curly brace
    match = re.search(r'\{.*\}', text, re.DOTALL)

    if not match:
        return {"error": "No JSON object found in response."}

    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": f"Failed to decode JSON: {json_str}"}


def safe_cast(val, to_type, default):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default