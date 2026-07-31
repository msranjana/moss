import asyncio
import inspect
from types import SimpleNamespace

from moss_cli.commands.playground import PLAYGROUND_HTML, PlaygroundHandler


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
        PlaygroundHandler._latest_request_id = 99
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
    monkeypatch.setattr(PlaygroundHandler, "_latest_request_id", 0)
    return handler, captured


def _valid_body():
    return {"name": "idx", "query": "hello", "requestId": 1, "topK": 5, "alpha": 0.5}


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
    monkeypatch.setattr(PlaygroundHandler, "_latest_request_id", 5)
    body = _valid_body()
    body["requestId"] = 3
    handler._handle_post_query(body)
    assert captured["status"] == 200
    assert captured["data"].get("superseded") is True
    assert client.query_calls == []


def test_query_skips_stale_queued_job(monkeypatch):
    client = FakeClient()
    handler, captured = _make_handler(
        monkeypatch, client=client, worker=SupersedingWorker(client)
    )
    handler._handle_post_query(dict(_valid_body()))
    assert captured["status"] == 200
    assert captured["data"].get("superseded") is True
    assert client.query_calls == []
