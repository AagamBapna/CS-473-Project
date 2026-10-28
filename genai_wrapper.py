"""Course text-only GenAI wrapper.

Student-facing use mirrors the common OpenAI chat-completions shape::

    from genai_wrapper import OpenAI
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="gpt-oss:120b",
        messages=[{"role": "user", "content": "..."}],
        temperature=0,
        max_tokens=1024,
    )
    text = response.choices[0].message.content

Only ordinary text chat is supported. There are intentionally no tool,
function, file, attachment, container, or arbitrary endpoint options.

Locally, this file talks directly to the course GenAI endpoint with the
student's key. Similar mechanism is used ono the autograder
"""
from __future__ import annotations

import copy
import json
import queue
import re
import socket
import sys
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


BASE_URL = "https://genai.rcac.purdue.edu/api"

MAX_FRAME_BYTES = 262144

def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value

def _invalid_number(value):
    raise ValueError("Non-finite JSON number")

def receive(stream):
    def read(n):
        parts = []
        while n:
            block = stream.recv(n)
            if not block:
                raise EOFError("Broker connection closed")
            parts.append(block)
            n -= len(block)
        return b"".join(parts)
    size = struct.unpack("!I", read(4))[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError("Broker message exceeds size limit")
    value = json.loads(read(size), object_pairs_hook=_object, parse_constant=_invalid_number)
    if not isinstance(value, dict):
        raise ValueError("Broker message must be an object")
    return value

def send(stream, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError("Broker response exceeds size limit")
    stream.sendall(struct.pack("!I", len(data)) + data)

ALLOWED_MODELS = {'gpt-oss:120b', 'llama4:latest', 'gemma4:26b-a4b', 'qwen3.8:27b', 'codellama:latest', 'devstral-small-2:latest', 'gemma3:1b', 'gemma3:12b', 'gemma3:27b', 'gpt-oss:latest', 'llama3.1:latest', 'llama3.1:70b-instruct-q4_K_M', 'llama3.2:latest', 'llama3.3:70b', 'llava:latest', 'medgemma:27b', 'mistral:latest', 'phi4:latest', 'qwen2.5:72b', 'qwen3:1.7b', 'qwen3:8b', 'qwen3:14b', 'qwen3:30b', 'qwen3-coder:latest', 'qwq:latest', 'qwq:32b-fp16'}
_BROKER = None
_CLIENTS = {}
_CLIENT_LOCK = threading.Lock()
_SECRETS = set()
_OUTPUT_LOCK = threading.Lock()


class GenAIError(RuntimeError):
    def __init__(self, message, *, http_status=None, server_detail=None, retry_smaller=False):
        super().__init__(message)
        self.http_status = http_status
        self.server_detail = server_detail
        self.retry_smaller = retry_smaller


def redact(value):
    text = str(value)
    for secret in _SECRETS:
        text = text.replace(secret, "[REDACTED_API_KEY]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", text)


def debug_log(prefix, message):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _OUTPUT_LOCK:
        print(f"[{prefix}][{stamp}] {redact(message)}", file=sys.stderr, flush=True)


def _clean_messages(messages):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages:
        raise GenAIError("messages must be a nonempty list of text chat messages")
    clean = []
    total = 0
    for item in messages:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise GenAIError("Each message must contain only role and content")
        role, content = item["role"], item["content"]
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise GenAIError("Only system/user/assistant text messages are supported")
        total += len(content)
        clean.append({"role": role, "content": content})
    if total <= 0 or total > 100_000:
        raise GenAIError("Text request is empty or exceeds 100000 characters")
    return clean


class _Connection:
    def __init__(self, fds, timeout):
        self.pool = queue.Queue()
        sockets = [socket.socket(fileno=fd) for fd in fds]
        if len(sockets) != 11:
            raise GenAIError("Invalid broker descriptor set")
        for stream in sockets:
            stream.settimeout(timeout)
        self.control = sockets[-1]
        self.control_lock = threading.Lock()
        for stream in sockets[:-1]:
            self.pool.put(stream)
        self.timeout = timeout

    def call(self, request, *, control=False):
        stream = self.control if control else self.pool.get(timeout=self.timeout)
        lock = self.control_lock if control else None
        try:
            if lock:
                lock.acquire()
            send(stream, request)
            return receive(stream)
        except (OSError, EOFError, ValueError) as exc:
            raise GenAIError("Trusted API broker connection failed: " + str(exc)) from None
        finally:
            if lock:
                lock.release()
            else:
                self.pool.put(stream)


def attach_broker(fds, *, timeout):
    global _BROKER
    _BROKER = _Connection(fds, timeout)
    with _CLIENT_LOCK:
        _CLIENTS.clear()


def configure_policy(*, allowed_models=None, **unused):
    global ALLOWED_MODELS
    if allowed_models is not None:
        ALLOWED_MODELS = set(allowed_models)


class _BrokerClient:
    def __init__(self, api_key=None):
        if _BROKER is None:
            raise GenAIError("Trusted grader broker is not attached")
        self._broker = _BROKER
        self._local = threading.local()
        if isinstance(api_key, str) and api_key:
            _SECRETS.add(api_key)

    @property
    def last_response(self):
        return copy.deepcopy(getattr(self._local, "metadata", {}))

    def _call(self, request, *, control=False):
        reply = self._broker.call(request, control=control)
        self._local.metadata = reply.get("metadata", {})
        if not reply.get("ok"):
            raise GenAIError(
                reply.get("error", "Broker rejected request"),
                http_status=reply.get("http_status"),
                server_detail=reply.get("server_detail"),
                retry_smaller=reply.get("retry_smaller", False),
            )
        return reply.get("value")

    def complete(self, model, messages, *, temperature=0.0, max_tokens=8192, request_label=None):
        return self._call({
            "op": "complete",
            "model": model,
            "messages": _clean_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "request_label": request_label,
        })

    def stats(self):
        return self._call({"op": "stats"}, control=True)

    def cancel(self):
        return self._call({"op": "cancel"}, control=True)

    def list_models(self):
        return self._call({"op": "list_models"})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _DirectClient:
    """Small local-only client; no tools/files/containers are implemented."""
    def __init__(self, api_key):
        if not isinstance(api_key, str) or not api_key.strip():
            raise GenAIError("API_KEY is empty")
        self._key = api_key.strip()
        _SECRETS.add(self._key)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._last_start = 0.0
        self._stats = {"logical_calls": 0, "attempts": 0, "successes": 0, "errors": 0, "rate_rpm": 20}

    @property
    def last_response(self):
        return copy.deepcopy(getattr(self._local, "metadata", {}))

    def _pace(self):
        with self._lock:
            delay = max(0.0, self._last_start + 3.1 - time.monotonic())
            if delay:
                time.sleep(delay)
            self._last_start = time.monotonic()
            self._stats["attempts"] += 1

    def _request(self, path, payload=None):
        self._pace()
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        req = urllib.request.Request(
            BASE_URL + path,
            data=data,
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.build_opener(_NoRedirect()).open(req, timeout=240) as response:
                raw = response.read(2_000_001)
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read(2_000_001)
            status = exc.code
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._stats["errors"] += 1
            raise GenAIError("Transport failure or request timeout", server_detail=redact(exc)) from None
        if len(raw) > 2_000_000:
            self._stats["errors"] += 1
            raise GenAIError("Response exceeded the permitted size")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self._stats["errors"] += 1
            raise GenAIError("The server returned invalid JSON", http_status=status) from None
        self._local.metadata = {"http_status": status, "model": body.get("model") if isinstance(body, dict) else None}
        if not 200 <= status < 300:
            self._stats["errors"] += 1
            raise GenAIError(f"GenAI HTTP {status}", http_status=status, server_detail=redact(body))
        return body

    def complete(self, model, messages, *, temperature=0.0, max_tokens=8192, request_label=None):
        if model not in ALLOWED_MODELS:
            raise GenAIError("Model is not enabled by the instructor")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 16384:
            raise GenAIError("max_tokens must be an integer from 1 to 16384")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= float(temperature) <= 2:
            raise GenAIError("temperature must be from 0 to 2")
        self._stats["logical_calls"] += 1
        body = self._request("/chat/completions", {
            "model": model,
            "messages": _clean_messages(messages),
            "temperature": float(temperature),
            "max_tokens": max_tokens,
            "stream": False,
        })
        try:
            message = body["choices"][0]["message"]
            if message.get("tool_calls") or message.get("function_call"):
                raise GenAIError("Tool/function responses are not supported")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError):
            self._stats["errors"] += 1
            raise GenAIError("No final text response was returned") from None
        self._stats["successes"] += 1
        return content

    def list_models(self):
        body = self._request("/models")
        records = body.get("data", []) if isinstance(body, dict) else []
        # Only expose course-enabled models that are actually available to this key.
        return [
            x["id"]
            for x in records
            if isinstance(x, dict)
            and isinstance(x.get("id"), str)
            and x["id"] in ALLOWED_MODELS
        ]

    def stats(self):
        return dict(self._stats)

    def cancel(self):
        return None


def get_client(api_key=None):
    mode = "broker" if _BROKER is not None else "direct"
    key = "broker" if mode == "broker" else (api_key or "")
    cache_key = (mode, key)
    with _CLIENT_LOCK:
        if cache_key not in _CLIENTS:
            _CLIENTS[cache_key] = _BrokerClient(api_key) if mode == "broker" else _DirectClient(api_key)
        return _CLIENTS[cache_key]


@dataclass(frozen=True)
class _Message:
    content: str
    role: str = "assistant"


@dataclass(frozen=True)
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass(frozen=True)
class _ChatCompletion:
    choices: list
    model: str


@dataclass(frozen=True)
class _ModelRecord:
    id: str


@dataclass(frozen=True)
class _ModelList:
    data: list


class _Completions:
    def __init__(self, backend):
        self._backend = backend

    def create(self, *, model, messages, temperature=0.0, max_tokens=8192):
        text = self._backend.complete(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _ChatCompletion(choices=[_Choice(_Message(text))], model=model)


class _Chat:
    def __init__(self, backend):
        self.completions = _Completions(backend)


class _Models:
    def __init__(self, backend):
        self._backend = backend

    def list(self):
        return _ModelList([_ModelRecord(model_id) for model_id in self._backend.list_models()])


class OpenAI:
    """Text-only course client with the familiar ``client.chat.completions`` shape."""
    def __init__(self, api_key=None):
        backend = get_client(api_key)
        self.chat = _Chat(backend)
        self.models = _Models(backend)


def generate(api_key, model, messages, *, temperature=0.0, max_tokens=8192):
    return get_client(api_key).complete(model, messages, temperature=temperature, max_tokens=max_tokens)


def get_stats(api_key=None):
    return get_client(api_key).stats()
