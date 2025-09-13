"""
Minimal helpers for Gemini chat, classification, and embeddings.
Inputs: env var `GEMINI_API_KEY`; model names and system prompts via args.
Outputs: embeddings (np.ndarray), classification dict, chat text.
Run: python chat_cli.py u1 "I love matcha latte"
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
# Minimal structured output schemas for Gemini SDK (avoid unsupported fields like title/default)
_MEMORY_OBJ_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "should_write": {"type": "BOOLEAN"},
        "memory_type": {"type": "STRING", "enum": ["preference", "profile", "fact", "other"]},
        "key": {"type": "STRING"},
        "value": {"type": "STRING"},
        "expires_at": {"type": "STRING", "nullable": True},
    },
    "required": ["should_write", "memory_type", "key", "value"],
}

_CLASSIFY_SCHEMA = {
    "type": "ARRAY",
    "items": _MEMORY_OBJ_SCHEMA,
}


def _model_name(name: str) -> str:
    return name if name.startswith("models/") else f"models/{name}"


class GeminiClient:
    def __init__(
        self,
        chat_model: str = "gemini-2.5-flash",
        classify_model: str = "gemini-2.5-flash-lite",
        embed_model: str = "gemini-embedding-exp-03-07",
        api_key_env: str = "GEMINI_API_KEY",
        chat_system: str = (
            "You are a helpful assistant that can answer questions and help with tasks."
        ),
        classify_system: str = (
            "You extract ZERO OR MORE user memories from a single message. Return ONLY a JSON array of objects; "
            "each object has keys: should_write (bool), memory_type (preference|profile|fact|other), key (snake_case, 1-4 words), "
            "value (<=60 chars, normalized), expires_at (YYYY-MM-DD or null). Save only if useful across future conversations. "
            "For ephemeral items (e.g., meetings, temporary status), set a near-term expires_at; otherwise leave null. "
            "Do NOT save sensitive data (passwords, phone, email). If nothing should be saved, return []."
        ),
    ) -> None:
        assert os.getenv(api_key_env), f"Missing {api_key_env}"
        import google.generativeai as genai  # type: ignore

        self._genai = genai
        self._genai.configure(api_key=os.environ[api_key_env])
        self._chat_model = self._genai.GenerativeModel(
            _model_name(chat_model), system_instruction=chat_system
        )
        self._classify_model = self._genai.GenerativeModel(
            _model_name(classify_model), system_instruction=classify_system
        )
        self._embed_model_name = _model_name(embed_model)
        self.messages = []

    def embed(self, text: str) -> np.ndarray:
        r = self._genai.embed_content(model=self._embed_model_name, content=text)
        v = np.array(r["embedding"], dtype=np.float32)
        n = np.linalg.norm(v)
        assert n > 0, "Zero embedding norm"
        return v / n

    def classify_memory(self, user_text: str) -> List[Dict[str, object]]:
        resp = self._classify_model.generate_content(
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_text},
                    ],
                }
            ],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": _CLASSIFY_SCHEMA,
                "temperature": 0,
            },
        )
        txt = resp.text.strip()
        obj = json.loads(txt)
        if isinstance(obj, dict):
            obj = [obj]
        return obj

    def extract_memories(self, transcript: str, max_items: int = 10) -> List[Dict[str, object]]:
        resp = self._classify_model.generate_content(
            [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "From the transcript below (role-labeled), extract at most "
                                f"{max_items} durable memories that would help future conversations with the same user. "
                                "Normalize keys to snake_case (1-4 words) and concise values (<=60 chars). "
                                "Exclude ephemeral details and sensitive data. Return ONLY a strict JSON array of objects with "
                                "fields: should_write (bool), memory_type (preference|profile|fact|other), key, value, "
                                "expires_at (YYYY-MM-DD or null). If nothing should be saved, return [].\n\n"
                                f"Transcript:\n{transcript}"
                            )
                        },
                    ],
                }
            ],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": _CLASSIFY_SCHEMA,
                "temperature": 0,
            },
        )
        txt = resp.text.strip()
        obj = json.loads(txt)
        if isinstance(obj, dict):
            obj = [obj]
        return obj

    def chat(self, user_text: str, memories: List[str]) -> str:
        context = "\n".join(f"- {m}" for m in memories) if memories else "(none)"
        prompt = (
            "Relevant memories (use only if helpful):\n"
            f"{context}\n\n"
            "User message:\n"
            f"{user_text}\n\n"
        )
        resp = self._chat_model.generate_content(
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                    ],
                }
            ]
        )
        return resp.text.strip()

    def chat_with_history(
        self, history: List[Dict[str, List[str]]], user_text: str, memories: List[str]
    ) -> str:
        context = "\n".join(f"- {m}" for m in memories) if memories else "(none)"
        user_payload = (
            "Relevant memories (use only if helpful):\n"
            f"{context}\n\n"
            "User message:\n"
            f"{user_text}\n\n"
        )
        contents = list(history) + [{"role": "user", "parts": [user_payload]}]
        resp = self._chat_model.generate_content(contents)
        return resp.text.strip()
