"""
Nova WebSocket + HTTP server.

  HTTP  :5001  — health check, message history, text input
  WS    :8766  — real-time state events + LLM token stream

WS message types (JSON):
  {"type": "state",   "state": "idle|listening|thinking|speaking"}
  {"type": "message", "role": "user|assistant", "content": "..."}
  {"type": "token",   "token": "..."}    ← streaming LLM tokens

Ports differ from Jarvis (3000/8765) so both can run simultaneously.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

log = logging.getLogger("nova.ws")


class NovaWSServer:
    def __init__(
        self,
        http_port: int,
        ws_port: int,
        on_text_message: Callable[[str], None],
    ) -> None:
        self.http_port        = http_port
        self.ws_port          = ws_port
        self.on_text_message  = on_text_message

        self._clients: set    = set()
        self._clients_lock    = threading.Lock()
        self._messages: list  = []           # rolling window for /api/messages
        self._state           = "idle"
        self._running         = False
        # Event loop that owns the WS connections. Broadcasts originate on other
        # threads (LLM worker, main), so sends must be scheduled back onto it.
        self._loop            = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._run_http, daemon=True, name="nova-http").start()
        threading.Thread(target=self._run_ws,   daemon=True, name="nova-ws").start()
        log.info(f"HTTP :{self.http_port}  WS :{self.ws_port}")

    def stop(self) -> None:
        self._running = False

    # ── Outbound helpers ──────────────────────────────────────────────────────────
    def broadcast_state(self, state: str) -> None:
        self._state = state
        self._ws_broadcast({"type": "state", "state": state})

    def send_message(self, role: str, content: str) -> None:
        msg = {"type": "message", "role": role, "content": content}
        self._messages.append(msg)
        if len(self._messages) > 200:
            self._messages = self._messages[-200:]
        self._ws_broadcast(msg)

    def stream_token(self, token: str) -> None:
        self._ws_broadcast({"type": "token", "token": token})

    def _ws_broadcast(self, payload: dict) -> None:
        # Called from non-async threads (LLM worker, main). websockets' send() is
        # a coroutine, so hand each send to the WS event loop rather than calling
        # it synchronously (which silently never transmits).
        loop = self._loop
        if loop is None:
            return
        data = json.dumps(payload)
        with self._clients_lock:
            clients = list(self._clients)

        async def _send_all():
            dead: set = set()
            for client in clients:
                try:
                    await client.send(data)
                except Exception:
                    dead.add(client)
            if dead:
                with self._clients_lock:
                    self._clients -= dead

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), loop)
        except Exception:
            pass

    # ── WebSocket server ──────────────────────────────────────────────────────────
    def _run_ws(self) -> None:
        try:
            import websockets

            server_ref = self

            async def handler(websocket):
                with server_ref._clients_lock:
                    server_ref._clients.add(websocket)
                # Send current state immediately on connect
                await websocket.send(
                    json.dumps({"type": "state", "state": server_ref._state})
                )
                try:
                    async for raw in websocket:
                        try:
                            data = json.loads(raw)
                            if data.get("type") == "message":
                                content = data.get("content", "").strip()
                                if content:
                                    server_ref.on_text_message(content)
                        except (json.JSONDecodeError, KeyError):
                            pass
                except Exception:
                    pass
                finally:
                    with server_ref._clients_lock:
                        server_ref._clients.discard(websocket)

            async def main():
                # Publish the running loop so _ws_broadcast can schedule sends.
                server_ref._loop = asyncio.get_running_loop()
                async with websockets.serve(handler, "localhost", self.ws_port):
                    await asyncio.Future()   # run forever

            asyncio.run(main())

        except ImportError:
            log.warning("websockets package not installed — WS server disabled.")
        except Exception as exc:
            log.error(f"WS server error: {exc}")

    # ── HTTP server ───────────────────────────────────────────────────────────────
    def _run_http(self) -> None:
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # suppress per-request logs

            def do_GET(self):
                if self.path == "/api/status":
                    self._json({"status": "ok", "state": server_ref._state})

                elif self.path.startswith("/api/messages"):
                    self._json({"messages": server_ref._messages[-50:]})

                else:
                    self.send_error(404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length)
                try:
                    data = json.loads(body)
                except Exception:
                    data = {}

                if self.path == "/api/message":
                    content = (data.get("content") or "").strip()
                    if content:
                        threading.Thread(
                            target=server_ref.on_text_message,
                            args=(content,),
                            daemon=True,
                        ).start()
                    self._json({"ok": True})

                elif self.path == "/api/mute":
                    # State managed by VoiceAssistant; this endpoint is for the Swift UI
                    self._json({"ok": True})

                else:
                    self.send_error(404)

            def do_OPTIONS(self):
                # Basic CORS for local dev tools
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def _json(self, payload: dict):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

        try:
            HTTPServer(("localhost", self.http_port), Handler).serve_forever()
        except Exception as exc:
            log.error(f"HTTP server error: {exc}")
