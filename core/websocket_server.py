"""
websocket_server.py — Servidor WebSocket Python ↔ Electron.

Protocolo resumido (ver docs/websocket_protocol.md para el detalle):

Python → Electron
    { "tipo": "frame", "gesto": str, "confianza": float,
      "mano": {"x": f, "y": f}, "objeto_activo": str|null,
      "estado_objeto": str, "slide_actual": int,
      "iluminacion_ok": bool, "mano_detectada": bool,
      "objetos": [ ...estado de todos los objetos del slide actual... ] }
    { "tipo": "calibracion_ok" }
    { "tipo": "luz_baja" }
    { "tipo": "objeto_soltado", ... }
    { "tipo": "hello", "version": "0.1.0", "mode": "ghost" }

Electron → Python
    { "tipo": "reiniciar" }
    { "tipo": "cambiar_slide", "slide": int }
    { "tipo": "finalizar" }
    { "tipo": "cargar_sesion", "path": str }
    { "tipo": "set_mirror", "value": bool }

Regla de resiliencia: ningún lado espera respuesta. Si el canal cae,
cada lado sigue con su último estado y se reconecta automáticamente.
"""

from __future__ import annotations

import asyncio
import json
import threading
from queue import Queue, Empty
from typing import Callable, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class WebSocketServer:
    """Servidor WebSocket que corre en su propio hilo con un loop asyncio.

    Otros hilos del backend hacen `.send(dict)` (no bloqueante). Los mensajes
    entrantes se entregan al callback `on_message(dict)` desde el hilo del loop.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        on_message: Optional[Callable[[dict], None]] = None,
        on_connect: Optional[Callable[[], list]] = None,
    ):
        self.host = host
        self.port = port
        self.on_message = on_message or (lambda _: None)
        # on_connect debe devolver una lista de dicts — se envían al cliente
        # recién conectado tras el `hello`. Útil para reenviar estado actual
        # (modo_gesto, error_camara, etc.) a clientes que se conectan tarde.
        self.on_connect = on_connect or (lambda: [])

        self._outbox: "Queue[dict]" = Queue()
        self._clients: Set[WebSocketServerProtocol] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # API pública (thread-safe)
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="gestdeck-ws")
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def send(self, message: dict) -> None:
        """Encola un mensaje para broadcast a todos los clientes conectados."""
        if not self._running:
            return
        self._outbox.put_nowait(message)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"[ws] Servidor WebSocket en ws://{self.host}:{self.port}")
            # Bucle de envío
            while self._running:
                await self._flush_outbox()
                await asyncio.sleep(1 / 120)   # 120 Hz drenaje de cola

    async def _shutdown(self) -> None:
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        try:
            await ws.send(json.dumps({"tipo": "hello", "version": "0.1.0", "mode": "ghost"}))
            # Re-emitir estado actual (modo_gesto, estado de cámara, etc.)
            try:
                for msg in self.on_connect() or []:
                    await ws.send(json.dumps(msg))
            except Exception as e:
                print(f"[ws] Error en on_connect: {e}")
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                try:
                    self.on_message(msg)
                except Exception as e:
                    print(f"[ws] Error en on_message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    async def _flush_outbox(self) -> None:
        """Envía todos los mensajes encolados a todos los clientes."""
        while True:
            try:
                msg = self._outbox.get_nowait()
            except Empty:
                return
            if not self._clients:
                continue
            data = json.dumps(msg)
            dead = []
            # Snapshot: otro coroutine (handshake de un cliente nuevo) puede
            # añadir a self._clients durante los `await ws.send(...)` de abajo.
            for ws in tuple(self._clients):
                try:
                    await ws.send(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    def on_msg(m): print("<-", m)
    srv = WebSocketServer(on_message=on_msg)
    srv.start()
    try:
        i = 0
        while True:
            srv.send({"tipo": "frame", "ping": i})
            i += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        srv.stop()
