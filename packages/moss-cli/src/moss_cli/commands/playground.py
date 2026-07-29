"""moss playground — local web UI for querying Moss indexes interactively.

Starts a local HTTP server that serves a browser-based playground.
Index and query operations are proxied through the server's credentialed
MossClient — the project key never reaches the browser.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

import typer
from rich.console import Console

from moss import MossClient, QueryOptions

from ..config import resolve_credentials

console = Console()
HERE = Path(__file__).resolve().parent.parent

PLAYGROUND_HTML = HERE / "playground" / "index.html"


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


class PlaygroundHandler(SimpleHTTPRequestHandler):
    """HTTP handler — serves the playground UI and proxies SDK calls server-side
    so the project key is never exposed to the browser."""

    client: MossClient | None = None
    _token: str = ""
    _server_host: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE / "playground"), **kwargs)

    def _check_api_request(self) -> bool:
        if self.headers.get("X-Moss-Token") != self._token:
            self._send_json(403, {"error": "Forbidden: invalid or missing token"})
            return False
        host = self.headers.get("Host", "")
        if host and host != self._server_host and host != self._server_host.replace("127.0.0.1", "localhost"):
            self._send_json(403, {"error": "Forbidden: invalid Host header"})
            return False
        origin = self.headers.get("Origin", "")
        if origin and origin != f"http://{self._server_host}":
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
        if client is None:
            self._send_json(500, {"error": "MossClient not initialized"})
            return
        try:
            indexes = asyncio.run(client.list_indexes())
            self._send_json(200, {"indexes": [_index_info_to_dict(i) for i in indexes]})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_get_index(self, name: str | None) -> None:
        client = self.client
        if client is None:
            self._send_json(500, {"error": "MossClient not initialized"})
            return
        if not name:
            self._send_json(400, {"error": "Missing 'name' query parameter"})
            return
        try:
            info = asyncio.run(client.get_index(name))
            self._send_json(200, {"index": _index_info_to_dict(info)})
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
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(body)

        if path == "/api/load-index":
            self._handle_post_load_index(data)
        elif path == "/api/query":
            self._handle_post_query(data)
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_post_load_index(self, data: dict) -> None:
        client = self.client
        if client is None:
            self._send_json(500, {"error": "MossClient not initialized"})
            return
        name = data.get("name", "")
        if not name:
            self._send_json(400, {"error": "Missing 'name' in request body"})
            return
        try:
            asyncio.run(client.load_index(name))
            self._send_json(200, {"name": name})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_post_query(self, data: dict) -> None:
        client = self.client
        if client is None:
            self._send_json(500, {"error": "MossClient not initialized"})
            return
        name = data.get("name", "")
        query = data.get("query", "")
        if not name or not query:
            self._send_json(400, {"error": "Missing 'name' or 'query' in request body"})
            return
        top_k = data.get("topK")
        alpha = data.get("alpha")
        try:
            opts = QueryOptions(top_k=top_k, alpha=alpha)
            result = asyncio.run(client.query(name, query, opts))
            docs = []
            for d in result.docs:
                docs.append({
                    "id": d.id,
                    "text": d.text,
                    "score": d.score,
                    "metadata": d.metadata,
                })
            self._send_json(200, {
                "docs": docs,
                "timeTakenMs": result.time_taken_ms,
                "query": result.query,
            })
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
    client = MossClient(pid, pkey)

    # Start server
    final_port = port if port else _find_free_port()
    server_addr = ("127.0.0.1", final_port)

    PlaygroundHandler.client = client
    PlaygroundHandler._token = secrets.token_urlsafe(32)
    PlaygroundHandler._server_host = f"127.0.0.1:{final_port}"

    server = HTTPServer(server_addr, PlaygroundHandler)
    url = f"http://127.0.0.1:{final_port}"
    frag_url = f"{url}/#{PlaygroundHandler._token}"

    console.print()
    console.print("  [bold]Moss Playground[/bold]")
    console.print(f"  [dim]Server:[/dim]  [cyan]{url}[/cyan]")
    console.print(f"  [dim]Project:[/dim] {pid[:8]}...")
    console.print("  [dim]Stop:[/dim]    Ctrl+C")
    console.print()

    if not no_open:
        webbrowser.open(frag_url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        server.server_close()