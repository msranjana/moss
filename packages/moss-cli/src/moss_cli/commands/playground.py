"""moss playground — local web UI for querying Moss indexes interactively.

Starts a local HTTP server that serves a browser-based playground.
Index and query operations are proxied through the server's credentialed
MossClient — the project key never reaches the browser.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import secrets
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse, parse_qs

import typer
from rich.console import Console

from moss import MossClient, QueryOptions

from ..config import resolve_credentials

console = Console()
HERE = Path(__file__).resolve().parent.parent

PLAYGROUND_HTML = HERE / "playground" / "index.html"


class AsyncWorker:
    """Single background thread with one persistent event loop.

    All MossClient calls are submitted here so they execute serially on
    one loop/thread — safe even when multiple HTTP handler threads are
    active (ThreadingHTTPServer).

    `submit` takes a zero-arg callable rather than an already-constructed
    coroutine. This ensures the coroutine object itself is *constructed*
    on the worker's event loop (inside `runner`), not on the calling
    (HTTP handler) thread. Some async SDK/PyO3 objects bind to "the
    currently running loop" at construction time, so constructing them
    off-thread and only awaiting them here would still be unsafe.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._lock = asyncio.run_coroutine_threadsafe(self._make_lock(), self._loop).result()

    async def _make_lock(self) -> asyncio.Lock:
        return asyncio.Lock()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro_fn: Callable[[], Any]) -> Any:
        """Run coro_fn() entirely on the worker loop/thread and return its result.

        coro_fn must be a zero-arg callable. It may return any awaitable
        object (coroutine, Future, or a custom __await__-based object)
        or a plain value — either is handled.
        """

        async def runner():
            async with self._lock:
                result = coro_fn()
                if inspect.isawaitable(result):
                    return await result
                return result

        future = asyncio.run_coroutine_threadsafe(runner(), self._loop)
        return future.result()

    def stop(self, timeout: float | None = 5.0) -> bool:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout)
        return self._thread.is_alive()


def _index_info_to_dict(info: Any) -> Dict[str, Any]:
    """Serialize an IndexInfo (Rust/PyO3 type) to a JSON-safe dict."""
    model = getattr(info, "model", None)
    model_dict = {"id": getattr(model, "id", None), "version": getattr(model, "version", None)} if model else None
    return {
        "id": getattr(info, "id", ""),
        "name": getattr(info, "name", ""),
        "version": getattr(info, "version", ""),
        "status": getattr(info, "status", ""),
        "docCount": getattr(info, "doc_count", 0),
        "createdAt": getattr(info, "created_at", ""),
        "updatedAt": getattr(info, "updated_at", ""),
        "model": model_dict,
    }


class DaemonThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer whose per-request threads are daemons.

    By default ThreadingHTTPServer's request threads are non-daemon, so
    server_close() blocks waiting for any in-flight request thread to
    finish. If a handler is parked in worker.submit(...) (e.g. a slow
    /api/query) when the user hits Ctrl+C, shutdown would hang before
    worker.stop() ever runs. Daemonizing request threads lets process
    exit proceed without waiting on them.
    """

    daemon_threads = True


class PlaygroundHandler(SimpleHTTPRequestHandler):
    """HTTP handler — serves the playground UI and proxies SDK calls server-side
    so the project key is never exposed to the browser."""

    client: MossClient | None = None
    _token: str = ""
    _server_host: str = ""
    _worker: AsyncWorker | None = None
    _latest_request_id: int = 0
    _request_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE / "playground"), **kwargs)

    def _check_api_request(self) -> bool:
        if self.headers.get("X-Moss-Token") != self._token:
            self._send_json(403, {"error": "Forbidden: invalid or missing token"})
            return False
        host = self.headers.get("Host", "")
        allowed_hosts = {self._server_host, self._server_host.replace("127.0.0.1", "localhost")}
        if host and host not in allowed_hosts:
            self._send_json(403, {"error": "Forbidden: invalid Host header"})
            return False
        origin = self.headers.get("Origin", "")
        allowed_origins = {f"http://{h}" for h in allowed_hosts}
        if origin and origin not in allowed_origins:
            self._send_json(403, {"error": "Forbidden: invalid Origin"})
            return False
        return True

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_index()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/indexes":
            if not self._check_api_request():
                return
            self._handle_list_indexes()
        elif path == "/api/index":
            if not self._check_api_request():
                return
            names = params.get("name", [])
            self._handle_get_index(names[0] if names else None)
        else:
            super().do_GET()

    def _serve_index(self) -> None:
        if not PLAYGROUND_HTML.exists():
            self._send_json(500, {"error": "Playground HTML not found"})
            return
        html = PLAYGROUND_HTML.read_text(encoding="utf-8")
        self._send_html(html)

    def _handle_list_indexes(self) -> None:
        client = self.client
        worker = self._worker
        if client is None or worker is None:
            self._send_json(500, {"error": "MossClient not initialized"})
            return
        try:
            async def _call():
                indexes = client.list_indexes()
                if inspect.isawaitable(indexes):
                    indexes = await indexes
                return [_index_info_to_dict(i) for i in indexes]
            indexes = worker.submit(_call)
            self._send_json(200, {"indexes": indexes})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_get_index(self, name: str | None) -> None:
        client = self.client
        worker = self._worker
        if client is None or worker is None:
            self._send_json(500, {"error": "MossClient not initialized"})
            return
        if not name:
            self._send_json(400, {"error": "Missing 'name' query parameter"})
            return
        try:
            async def _call():
                result = client.get_index(name)
                if inspect.isawaitable(result):
                    result = await result
                return _index_info_to_dict(result)
            info = worker.submit(_call)
            self._send_json(200, {"index": info})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, format, *args):
        if len(args) >= 3:
            console.print(f"  [dim]{args[0]} {args[1]} {args[2]}[/dim]")
        elif len(args) >= 2:
            console.print(f"  [dim]{args[0]} {args[1]}[/dim]")
        elif len(args) >= 1:
            console.print(f"  [dim]{args[0]}[/dim]")

    def do_POST(self) -> None:
        if not self._check_api_request():
            return
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"

        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        if not isinstance(data, dict):
            self._send_json(400, {"error": "Request body must be a JSON object"})
            return

        if path == "/api/load-index":
            self._handle_post_load_index(data)
        elif path == "/api/query":
            self._handle_post_query(data)
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_post_load_index(self, data: dict) -> None:
        client = self.client
        worker = self._worker
        if client is None or worker is None:
            self._send_json(500, {"error": "MossClient not initialized"})
            return
        name = data.get("name", "")
        if not name:
            self._send_json(400, {"error": "Missing 'name' in request body"})
            return
        try:
            worker.submit(lambda: client.load_index(name))
            self._send_json(200, {"name": name})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_post_query(self, data: dict) -> None:
        client = self.client
        worker = self._worker
        if client is None or worker is None:
            self._send_json(500, {"error": "MossClient not initialized"})
            return
        name = data.get("name", "")
        query = data.get("query", "")
        if not name or not query:
            self._send_json(400, {"error": "Missing 'name' or 'query' in request body"})
            return

        raw_request_id = data.get("requestId")
        if isinstance(raw_request_id, bool) or not isinstance(raw_request_id, int):
            self._send_json(400, {"error": "Missing or invalid 'requestId'"})
            return

        raw_top_k = data.get("topK")
        if raw_top_k is not None:
            if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
                self._send_json(400, {"error": "'topK' must be an integer"})
                return
            top_k = raw_top_k
        else:
            top_k = None

        raw_alpha = data.get("alpha")
        if raw_alpha is not None:
            if isinstance(raw_alpha, bool) or not isinstance(raw_alpha, (int, float)):
                self._send_json(400, {"error": "'alpha' must be a number"})
                return
            alpha = float(raw_alpha)
            if not math.isfinite(alpha):
                self._send_json(400, {"error": "'alpha' must be a finite number"})
                return
        else:
            alpha = None

        if top_k is not None and (top_k < 1 or top_k > 50):
            self._send_json(400, {"error": "topK must be between 1 and 50"})
            return
        if alpha is not None and (alpha < 0 or alpha > 1):
            self._send_json(400, {"error": "alpha must be between 0 and 1"})
            return

        with PlaygroundHandler._request_lock:
            if raw_request_id < PlaygroundHandler._latest_request_id:
                self._send_json(200, {"docs": [], "timeTakenMs": 0, "query": query, "superseded": True})
                return
            PlaygroundHandler._latest_request_id = raw_request_id

        try:
            def _build_result(r):
                return {
                    "docs": [
                        {"id": d.id, "text": d.text, "score": d.score, "metadata": d.metadata}
                        for d in r.docs
                    ],
                    "timeTakenMs": r.time_taken_ms,
                    "query": r.query,
                }
            async def _call():
                if raw_request_id < PlaygroundHandler._latest_request_id:
                    return {"docs": [], "timeTakenMs": 0, "query": query, "superseded": True}
                result = client.query(name, query, QueryOptions(top_k=top_k, alpha=alpha))
                if inspect.isawaitable(result):
                    result = await result
                return _build_result(result)
            payload = worker.submit(_call)
            self._send_json(200, payload)
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def _find_free_port(start: int = 8765, max_attempts: int = 20) -> int:
    for port in range(start, start + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find a free port in range {start}-{start + max_attempts}")


def playground_command(
    ctx: typer.Context,
    port: int = typer.Option(0, "--port", "-p", help="Port for the HTTP server (0 = auto)"),
    profile: Optional[str] = typer.Option(
        None, "--profile", help="Credential profile name",
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Do not open the browser automatically",
    ),
) -> None:
    """Start the Moss Playground — a local web UI for interactive search.

    Launches a browser-based playground where you can list cloud indexes,
    load one into server memory, and run queries against it.
    """
    if profile:
        ctx.obj["profile"] = profile

    # Resolve credentials
    pid, pkey = resolve_credentials(
        ctx.obj.get("project_id"), ctx.obj.get("project_key"), ctx.obj.get("profile")
    )

    if not pid or not pkey:
        console.print(
            "[red]No credentials found.[/red] Run [bold]moss init[/bold] first "
            "or set MOSS_PROJECT_ID and MOSS_PROJECT_KEY."
        )
        raise typer.Exit(1)

    # Initialize the SDK client for API endpoints
    worker = AsyncWorker()
    client = worker.submit(lambda: MossClient(pid, pkey))
    
    # Start server
    final_port = port if port else _find_free_port()
    server_addr = ("127.0.0.1", final_port)

    PlaygroundHandler.client = client
    PlaygroundHandler._worker = worker
    PlaygroundHandler._token = secrets.token_urlsafe(32)
    PlaygroundHandler._server_host = f"127.0.0.1:{final_port}"

    server = DaemonThreadingHTTPServer(server_addr, PlaygroundHandler)
    url = f"http://127.0.0.1:{final_port}"
    frag_url = f"{url}/#{PlaygroundHandler._token}"

    console.print()
    console.print("  [bold]Moss Playground[/bold]")
    console.print(f"  [dim]Server:[/dim]  [cyan]{frag_url}[/cyan]")
    console.print(f"  [dim]Project:[/dim] {pid[:8]}...")
    console.print("  [dim]Stop:[/dim]    Ctrl+C")
    console.print()

    opened = False
    if not no_open:
        opened = webbrowser.open(frag_url)
    if not opened:
        console.print("  [yellow]Open this URL in your browser:[/yellow]")
        console.print(f"  [cyan]{frag_url}[/cyan]")
        console.print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        server.server_close()
        if worker.stop():
            console.print("[yellow]Worker thread did not shut down gracefully within timeout[/yellow]")