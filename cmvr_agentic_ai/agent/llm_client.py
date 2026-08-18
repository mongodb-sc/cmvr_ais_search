"""Thin httpx wrapper for the Grove-gateway-proxied Anthropic Messages API.

This is the ONLY module in the project that talks to the LLM. It deliberately
does not use the `anthropic` SDK so the request shape matches the gateway's
curl contract exactly:
  - auth header is `api-key` (not `x-api-key` / `Authorization: Bearer`)
  - POST {GROVE_BASE_URL}
  - `anthropic-version: 2023-06-01` on every request
  - model string comes from GROVE_MODEL (default claude-fable-5)

The API key is read at call time (not import time) so importing this module
never crashes offline / index-only flows.
"""

from __future__ import annotations

import os

import httpx

import config


def call_claude(
    messages: list[dict],
    tools: list[dict] | None = None,
    system: str | None = None,
    max_tokens: int = 2000,
) -> dict:
    """Call the Grove gateway and return the raw parsed JSON response dict."""
    api_key = os.environ.get("GROVE_API_KEY")
    if not api_key:
        raise RuntimeError("GROVE_API_KEY is not set")

    payload: dict = {
        "model": config.GROVE_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    if system:
        payload["system"] = system

    response = httpx.post(
        config.GROVE_BASE_URL,
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "api-key": api_key,
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()
