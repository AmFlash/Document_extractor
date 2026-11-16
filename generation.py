
"""Structured Gemini API calls for tender-field extraction."""
import json
import os
import re
import time
from urllib.parse import quote

import requests


DEFAULT_MODEL = "gemini-3.6-flash"

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "name_of_work": {"type": "string"},
        "ato_criteria": {"type": "string"},
        "similar_work_definition": {"type": "string"},
        "documents_to_be_submitted": {"type": "array", "items": {"type": "string"}},
        "tech_evaluation_mode": {"type": "string", "enum": ["L1", "QCBS", "UNKNOWN"]},
    },
    "required": ["name_of_work", "ato_criteria", "similar_work_definition",
                 "documents_to_be_submitted", "tech_evaluation_mode"],
}


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No valid JSON object found in Gemini response")
    return json.loads(text[start:end + 1])


def _gemini_schema(schema):
    result = {}
    for key, value in schema.items():
        if key == "type":
            result[key] = value.upper()
        elif key == "items":
            result[key] = _gemini_schema(value)
        elif key == "properties":
            result[key] = {name: _gemini_schema(spec) for name, spec in value.items()}
        else:
            result[key] = value
    return result


def _call_gemini(system, user, model, api_key):
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(model, safe='')}:generateContent"
    )
    response = requests.post(
        endpoint,
        headers={"x-goog-api-key": api_key},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                "responseSchema": _gemini_schema(LLM_SCHEMA),
            },
        },
        timeout=1200,
    )
    response.raise_for_status()
    candidates = response.json().get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts)
    if not text:
        raise ValueError("Gemini returned no text in its candidate")
    return parse_json(text)


def ask_llm(system_prompt, user, model):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key is missing. Add GEMINI_API_KEY to the project .env file.")

    last = None
    for attempt in range(3):
        try:
            return _call_gemini(system_prompt, user, model, api_key)
        except Exception as error:
            last = error
            time.sleep(2 * (2 ** attempt))
    raise last
