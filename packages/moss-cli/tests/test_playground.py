import asyncio
import inspect
import io
from types import SimpleNamespace

from moss_cli.commands.playground import (
    MAX_REQUEST_BODY_SIZE,
    PLAYGROUND_HTML,
    PlaygroundHandler,
)


class FakeClient:
    def __init__(self):
        self.query_calls = []

    def query(self, name, query, options):
        self.query_calls.append((name, query, options))
        doc = SimpleNamespace(id="d1", text="hello", score=0.9, metadata={"k": "v"})
        return SimpleNamespace(docs=[doc], time_taken_ms=12, query=query)


class FakeWorker:
    def __init__(self, client):
        self.client = client

    def submit(self, coro_fn):
        result = coro_fn()
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result


class SupersedingWorker:
    """Simulates a newer query arriving while an older one is queued."""

    def __init__(self, client):
        self.client = client

    def submit(self, coro_fn):
        PlaygroundHandler._latest_request_ids["sess"] = 99
        result = coro_fn()
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result


def _make_handler(monkeypatch, client=None, worker=None):
    handler = PlaygroundHandler.__new__(PlaygroundHandler)
    captured = {}

    def fake_send_json(status, data):
        captured["status"] = status
        captured["data"] = data

    monkeypatch.setattr(handler, "_send_json", fake_send_json)
    monkeypatch.setattr(PlaygroundHandler, "client", client)
    monkeypatch.setattr(PlaygroundHandler, "_worker", worker)
    monkeypatch.setattr(PlaygroundHandler, "_latest_request_ids", {})
    return handler, captured


def _make_token_handler(monkeypatch):
    handler = PlaygroundHandler.__new__(PlaygroundHandler)
    captured = {}

    def fake_send_json(status, data):
        captured["status"] = status
        captured["data"] = data

    monkeypatch.setattr(handler, "_send_json", fake_send_json)
    monkeypatch.setattr(PlaygroundHandler, "_token", "secret-token")
    monkeypatch.setattr(PlaygroundHandler, "_server_host", "127.0.0.1:8765")
    return handler, captured


def _valid_body():
    return {
        "name": "idx",
        "query": "hello",
        "sessionId": "sess",
        "requestId": 1,
        "topK": 5,
        "alpha": 0.5,
    }


def _auth_headers(**extra):
    headers = {
        "X-Moss-Token": "secret-token",
        "Host": "127.0.0.1:8765",
        "Origin": "http://127.0.0.1:8765",
    }
    headers.update(extra)
    return headers


def test_playground_html_asset_exists():
    assert PLAYGROUND_HTML.exists(), (
        f"Playground HTML not found at {PLAYGROUND_HTML}. "
        "Check that the asset is included in the package data."
    )


def test_html_renders_metadata_safely():
    html = PLAYGROUND_HTML.read_text(encoding="utf-8")
    assert "result-metadata" in html
    assert "JSON.stringify(doc.metadata, null, 2)" in html
    assert "textContent" in html


def test_html_aborts_stale_queries():
    html = PLAYGROUND_HTML.read_text(encoding="utf-8")
    assert "AbortController" in html
    assert "signal: thisAbort.signal" in html
    assert "requestId: thisRequestId" in html


def test_query_accepts_valid_params(monkeypatch):
    client = FakeClient()
    worker = FakeWorker(client)
    handler, captured = _make_handler(monkeypatch, client=client, worker=worker)
    handler._handle_post_query(dict(_valid_body()))
    assert captured["status"] == 200
    assert len(client.query_calls) == 1
    name, query, options = client.query_calls[0]
    assert (name, query) == ("idx", "hello")
    assert options.top_k == 5
    assert options.alpha == 0.5
    assert captured["data"]["docs"][0]["metadata"] == {"k": "v"}


def test_query_rejects_bool_topk(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    body = _valid_body()
    body["topK"] = True
    handler._handle_post_query(body)
    assert captured["status"] == 400
    assert client.query_calls == []


def test_query_rejects_fractional_topk(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    body = _valid_body()
    body["topK"] = 1.9
    handler._handle_post_query(body)
    assert captured["status"] == 400
    assert client.query_calls == []


def test_query_rejects_string_topk(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    body = _valid_body()
    body["topK"] = "5"
    handler._handle_post_query(body)
    assert captured["status"] == 400
    assert client.query_calls == []


def test_query_rejects_out_of_range_topk(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    for bad in (0, 51):
        body = _valid_body()
        body["topK"] = bad
        handler._handle_post_query(body)
        assert captured["status"] == 400, f"expected 400 for topK={bad}"
        assert client.query_calls == []


def test_query_rejects_bool_alpha(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    body = _valid_body()
    body["alpha"] = True
    handler._handle_post_query(body)
    assert captured["status"] == 400
    assert client.query_calls == []


def test_query_rejects_non_finite_alpha(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    for bad in (float("nan"), float("inf"), float("-inf")):
        body = _valid_body()
        body["alpha"] = bad
        handler._handle_post_query(body)
        assert captured["status"] == 400, f"expected 400 for alpha={bad}"
        assert client.query_calls == []


def test_query_rejects_out_of_range_alpha(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    for bad in (-0.1, 1.1):
        body = _valid_body()
        body["alpha"] = bad
        handler._handle_post_query(body)
        assert captured["status"] == 400, f"expected 400 for alpha={bad}"
        assert client.query_calls == []


def test_query_requires_request_id(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    body = _valid_body()
    for bad in (None, True, "1"):
        body["requestId"] = bad
        handler._handle_post_query(body)
        assert captured["status"] == 400, f"expected 400 for requestId={bad}"
        assert client.query_calls == []


def test_query_drops_already_superseded_request(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    PlaygroundHandler._latest_request_ids["sess"] = 5
    body = _valid_body()
    body["requestId"] = 3
    handler._handle_post_query(body)
    assert captured["status"] == 200
    assert captured["data"].get("superseded") is True
    assert client.query_calls == []


def test_query_sessions_are_isolated(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    body = _valid_body()
    body["requestId"] = 1
    body["sessionId"] = "sess-a"
    handler._handle_post_query(body)
    assert captured["status"] == 200
    body["sessionId"] = "sess-b"
    handler._handle_post_query(body)
    assert captured["status"] == 200
    assert len(client.query_calls) == 2


def test_query_requires_session_id(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    for bad in (None, True, 123, ""):
        body = _valid_body()
        body["sessionId"] = bad
        handler._handle_post_query(body)
        assert captured["status"] == 400, f"expected 400 for sessionId={bad!r}"
        assert client.query_calls == []


def test_query_rejects_non_string_name_and_query(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(monkeypatch, client=client, worker=FakeWorker(client))
    for key in ("name", "query"):
        for bad in (123, "   "):
            body = _valid_body()
            body[key] = bad
            handler._handle_post_query(body)
            assert captured["status"] == 400, f"expected 400 for {key}={bad!r}"
            assert client.query_calls == []


def test_check_api_request_missing_token(monkeypatch):
    handler, captured = _make_token_handler(monkeypatch)
    handler.headers = {"Host": "127.0.0.1:8765"}
    assert handler._check_api_request() is False
    assert captured["status"] == 403


def test_check_api_request_wrong_token(monkeypatch):
    handler, captured = _make_token_handler(monkeypatch)
    handler.headers = {"X-Moss-Token": "wrong", "Host": "127.0.0.1:8765"}
    assert handler._check_api_request() is False
    assert captured["status"] == 403


def test_check_api_request_foreign_host(monkeypatch):
    handler, captured = _make_token_handler(monkeypatch)
    handler.headers = {"X-Moss-Token": "secret-token", "Host": "evil.example.com"}
    assert handler._check_api_request() is False
    assert captured["status"] == 403


def test_check_api_request_foreign_origin(monkeypatch):
    handler, captured = _make_token_handler(monkeypatch)
    handler.headers = {
        "X-Moss-Token": "secret-token",
        "Host": "127.0.0.1:8765",
        "Origin": "http://evil.example.com",
    }
    assert handler._check_api_request() is False
    assert captured["status"] == 403


def test_check_api_request_allows_valid_request(monkeypatch):
    handler, captured = _make_token_handler(monkeypatch)
    handler.headers = {
        "X-Moss-Token": "secret-token",
        "Host": "localhost:8765",
        "Origin": "http://localhost:8765",
    }
    assert handler._check_api_request() is True
    assert "status" not in captured


def _make_authed_handler(monkeypatch, client=None, worker=None):
    handler, captured = _make_handler(monkeypatch, client=client, worker=worker)
    monkeypatch.setattr(PlaygroundHandler, "_token", "secret-token")
    monkeypatch.setattr(PlaygroundHandler, "_server_host", "127.0.0.1:8765")
    return handler, captured


def test_post_rejects_malformed_content_length(monkeypatch):
    handler, captured = _make_authed_handler(monkeypatch)
    handler.path = "/api/query"
    handler.headers = _auth_headers(**{"Content-Length": "abc"})
    handler.rfile = io.BytesIO(b"{}")
    handler.do_POST()
    assert captured["status"] == 400


def test_post_rejects_negative_content_length(monkeypatch):
    handler, captured = _make_authed_handler(monkeypatch)
    handler.path = "/api/query"
    handler.headers = _auth_headers(**{"Content-Length": "-1"})
    handler.rfile = io.BytesIO(b"{}")
    handler.do_POST()
    assert captured["status"] == 400


def test_post_rejects_oversized_body(monkeypatch):
    handler, captured = _make_authed_handler(monkeypatch)
    handler.path = "/api/query"
    handler.headers = _auth_headers(**{"Content-Length": str(MAX_REQUEST_BODY_SIZE + 1)})
    handler.rfile = io.BytesIO(b"{}")
    handler.do_POST()
    assert captured["status"] == 413


def test_post_accepts_empty_body_fallback(monkeypatch):
    handler, captured = _make_authed_handler(
        monkeypatch, client=FakeClient(), worker=FakeWorker(FakeClient())
    )
    handler.path = "/api/query"
    handler.headers = _auth_headers()
    handler.rfile = io.BytesIO(b"{}")
    handler.do_POST()
    assert captured["status"] == 400


def test_query_skips_stale_queued_job(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(
        monkeypatch, client=client, worker=SupersedingWorker(client)
    )
    handler._handle_post_query(dict(_valid_body()))
    assert captured["status"] == 200
    assert captured["data"].get("superseded") is True
    assert client.query_calls == []
