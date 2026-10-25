"""Minimal client for a local Ollama server, using only the standard library."""

import json
import urllib.error
import urllib.request
from typing import Any

from jarvis.brain import BrainOffline, BrainTimeout

# Ignore HTTP(S)_PROXY settings: the server is local.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float,
        keep_alive: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._keep_alive = keep_alive
        self._options = {"num_predict": max_tokens, "temperature": temperature}

    def complete(self, system: str, user: str) -> str:
        """Ask the model; returns its raw (JSON-mode) reply text."""
        payload = self._post(
            "/api/chat",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "keep_alive": self._keep_alive,  # stay loaded between requests
                "options": self._options,
            },
            self._timeout,
        )
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        return content if isinstance(content, str) else ""

    def warm_up(self, timeout_seconds: float) -> None:
        """Load the model into memory now, so the first question is fast."""
        self._post("/api/generate", {"model": self._model, "keep_alive": self._keep_alive}, timeout_seconds)

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(
            self._base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _OPENER.open(request, timeout=timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:  # server is up but refused, e.g. model not pulled
            detail = error.read().decode("utf-8", "replace")[:200]
            raise BrainOffline(f"Ollama returned HTTP {error.code}: {detail}") from error
        except TimeoutError as error:  # connected, but the model was too slow to reply
            raise BrainTimeout(f"Ollama didn't answer within {timeout:g} s") from error
        except urllib.error.URLError as error:  # couldn't connect (refused or timed out)
            raise BrainOffline(f"Ollama is not reachable at {self._base_url}: {error.reason}") from error
        except (OSError, ValueError) as error:  # connection dropped, or not JSON
            raise BrainOffline(f"bad response from Ollama: {error}") from error
        if not isinstance(payload, dict):
            raise BrainOffline("bad response from Ollama: not a JSON object")
        return payload
