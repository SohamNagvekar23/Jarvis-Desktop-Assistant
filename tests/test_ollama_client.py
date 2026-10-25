"""OllamaClient against a throwaway local HTTP server (no real Ollama needed)."""

import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from jarvis.brain import BrainOffline, BrainTimeout
from jarvis.ollama_client import OllamaClient


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.status = 200
        self.body: bytes = json.dumps({"message": {"role": "assistant", "content": '{"answer": "Hi."}'}}).encode()
        self.delay = 0.0


@pytest.fixture
def server() -> Iterator[tuple[FakeOllama, str]]:
    fake = FakeOllama()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers["Content-Length"])
            fake.requests.append((self.path, json.loads(self.rfile.read(length))))
            time.sleep(fake.delay)
            try:
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(fake.body)
            except ConnectionError:  # the client already gave up (timeout test)
                pass

        def log_message(self, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield fake, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def client(url: str, timeout: float = 5.0) -> OllamaClient:
    return OllamaClient(url, "llama3.2:3b", timeout, "30m", max_tokens=150, temperature=0.2)


def test_chat_request_and_reply(server: tuple[FakeOllama, str]) -> None:
    fake, url = server
    assert client(url).complete("SYSTEM", "what time is it") == '{"answer": "Hi."}'
    path, body = fake.requests[0]
    assert path == "/api/chat"
    assert body == {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "what time is it"},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"num_predict": 150, "temperature": 0.2},
    }


def test_warm_up_loads_the_model_and_keeps_it(server: tuple[FakeOllama, str]) -> None:
    fake, url = server
    fake.body = b'{"model": "llama3.2:3b", "done": true}'
    client(url).warm_up(timeout_seconds=5)
    assert fake.requests == [("/api/generate", {"model": "llama3.2:3b", "keep_alive": "30m"})]


def test_ignores_proxy_settings(server: tuple[FakeOllama, str], monkeypatch: pytest.MonkeyPatch) -> None:
    fake, url = server
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # would fail if used
    monkeypatch.setenv("NO_PROXY", "")
    assert client(url).complete("s", "u") == '{"answer": "Hi."}'


@pytest.mark.parametrize(
    "body",
    [b'{"message": {}}', b'{"message": "hi"}', b'{"done": true}', b'{"message": {"content": 5}}'],
)
def test_missing_content_is_empty(server: tuple[FakeOllama, str], body: bytes) -> None:
    fake, url = server
    fake.body = body
    assert client(url).complete("s", "u") == ""  # parse_reply then rejects it


def test_model_not_pulled_is_offline(server: tuple[FakeOllama, str]) -> None:
    fake, url = server
    fake.status = 404
    fake.body = b'{"error": "model \\"llama3.2:3b\\" not found, try pulling it first"}'
    with pytest.raises(BrainOffline, match="HTTP 404.*not found"):
        client(url).complete("s", "u")


@pytest.mark.parametrize("body", [b"<html>oops</html>", b"[1, 2]"])
def test_garbage_response_is_offline(server: tuple[FakeOllama, str], body: bytes) -> None:
    fake, url = server
    fake.body = body
    with pytest.raises(BrainOffline, match="bad response"):
        client(url).complete("s", "u")


def test_slow_server_times_out(server: tuple[FakeOllama, str]) -> None:
    fake, url = server
    fake.delay = 1.0
    started = time.monotonic()
    with pytest.raises(BrainTimeout):
        client(url, timeout=0.2).complete("s", "u")
    assert time.monotonic() - started < 0.9


@pytest.mark.parametrize("timeout", [0.5, 5.0], ids=["connect-times-out", "connect-refused"])
def test_nothing_listening_is_offline_not_slow(timeout: float) -> None:
    # Windows retries refused localhost connections for ~2 s, so a short
    # timeout expires while connecting; that still means "offline".
    with socket.socket() as sock:  # grab a free port, then close it
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(BrainOffline, match="not reachable"):
        client(f"http://127.0.0.1:{port}", timeout=timeout).complete("s", "u")
